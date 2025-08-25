#!/bin/bash
#
# Build script for JARVIS Native C++ Extensions
# Builds both Fast Capture and Vision ML Router modules
# Usage: ./build.sh [options]
#   Options:
#     clean      - Clean all builds
#     test       - Run tests after building
#     capture    - Build only Fast Capture
#     vision     - Build only Vision ML
#     all        - Build all extensions (default)
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 JARVIS Native Extensions Build Script${NC}"
echo "=================================="

# Parse command line arguments
BUILD_CAPTURE=true
BUILD_VISION=true
RUN_TESTS=false
CLEAN_ONLY=false

if [ "$1" == "clean" ]; then
    CLEAN_ONLY=true
elif [ "$1" == "capture" ]; then
    BUILD_VISION=false
    echo "Building: Fast Capture only"
elif [ "$1" == "vision" ]; then
    BUILD_CAPTURE=false
    echo "Building: Vision ML Router only"
elif [ "$1" == "all" ] || [ -z "$1" ]; then
    echo "Building: Fast Capture + Vision ML Router"
else
    echo "Building: Fast Capture + Vision ML Router"
fi

if [ "$1" == "test" ] || [ "$2" == "test" ]; then
    RUN_TESTS=true
fi

# Check dependencies
echo -e "\n${YELLOW}Checking dependencies...${NC}"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}❌ CMake not found. Please install CMake first.${NC}"
    echo "   On macOS: brew install cmake"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found.${NC}"
    exit 1
fi

# Check for pybind11
echo "Checking for pybind11..."
if ! python3 -c "import pybind11" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  pybind11 not found. Installing...${NC}"
    pip3 install pybind11
fi

# Clean builds if requested
if [ "$CLEAN_ONLY" = true ] || [ "$1" == "--clean" ]; then
    echo -e "\n${YELLOW}Cleaning all builds...${NC}"
    rm -rf build/
    rm -rf dist/
    rm -rf *.egg-info
    rm -f *.so
    rm -f *.dylib
    rm -f vision_ml_router*.so vision_ml_router*.dylib
    echo -e "${GREEN}✓ All extensions cleaned${NC}"
    
    if [ "$CLEAN_ONLY" = true ]; then
        exit 0
    fi
fi

# Build Fast Capture if requested
if [ "$BUILD_CAPTURE" = true ]; then
    echo -e "\n${YELLOW}Building Fast Capture Extension...${NC}"
    echo "================================"
    
    # Create build directory
    echo -e "${YELLOW}Creating build directory...${NC}"
    mkdir -p build
    cd build

    # Configure with CMake
    echo -e "${YELLOW}Configuring with CMake...${NC}"
    cmake .. -DCMAKE_BUILD_TYPE=Release

    # Build
    echo -e "${YELLOW}Building C++ extension...${NC}"
    make -j$(sysctl -n hw.ncpu)

    # Install
    echo -e "${YELLOW}Installing extension...${NC}"
    make install

    # Go back to native_extensions directory
    cd ..

    # Test import
    echo -e "${YELLOW}Testing Fast Capture import...${NC}"
    if python3 -c "import fast_capture; print('✓ Fast Capture version:', fast_capture.VERSION)" 2>/dev/null; then
        echo -e "${GREEN}✓ Fast Capture built successfully!${NC}"
        CAPTURE_SUCCESS=true
    else
        echo -e "${RED}❌ Fast Capture import failed${NC}"
        CAPTURE_SUCCESS=false
    fi
else
    CAPTURE_SUCCESS=skipped
fi

# Run performance test if requested
if [ "$2" == "test" ] || [ "$1" == "test" ]; then
    echo -e "\n${YELLOW}Running performance test...${NC}"
    cd ../vision
    python3 test_enhanced_vision.py
fi

# Build Vision ML Router if requested
if [ "$BUILD_VISION" = true ]; then
    echo -e "\n${YELLOW}Building Vision ML Router...${NC}"
    echo "================================"

    # Save current directory
    ORIGINAL_DIR=$(pwd)

    # Check if setup_vision_ml.py exists
    if [ -f "setup_vision_ml.py" ]; then
        # Clean previous vision ML builds
        rm -f vision_ml_router*.so vision_ml_router*.dylib 2>/dev/null
        
        # Build vision ML extension
        echo -e "${YELLOW}Compiling C++ Vision ML Router...${NC}"
        python3 setup_vision_ml.py build_ext --inplace
        
        # Check if build succeeded
        if ls vision_ml_router*.so 1> /dev/null 2>&1 || ls vision_ml_router*.dylib 1> /dev/null 2>&1; then
            echo -e "${GREEN}✓ Vision ML Router compiled successfully!${NC}"
            
            # Copy to parent directory
            cp vision_ml_router*.so ../. 2>/dev/null || cp vision_ml_router*.dylib ../. 2>/dev/null
            
            # Test vision ML import
            echo -e "${YELLOW}Testing Vision ML Router...${NC}"
            if python3 -c "import vision_ml_router; score, action = vision_ml_router.analyze('describe my screen'); print('✓ Vision ML Router working! Score:', score, 'Action:', action)" 2>/dev/null; then
                echo -e "${GREEN}✓ Vision ML Router built successfully!${NC}"
                VISION_ML_SUCCESS=true
            else
                echo -e "${YELLOW}⚠️  Vision ML import failed (will use Python fallback)${NC}"
                VISION_ML_SUCCESS=false
            fi
        else
            echo -e "${YELLOW}⚠️  Vision ML build failed (will use Python fallback)${NC}"
            VISION_ML_SUCCESS=false
        fi
    else
        echo -e "${YELLOW}⚠️  Vision ML setup script not found${NC}"
        VISION_ML_SUCCESS=false
    fi

    # Return to original directory
    cd "$ORIGINAL_DIR"
else
    VISION_ML_SUCCESS=skipped
fi

echo -e "\n${GREEN}✅ Build Summary${NC}"
echo -e "=================="

# Display build results
echo -e "\nExtension Status:"

# Fast Capture status
if [ "$CAPTURE_SUCCESS" = true ]; then
    echo -e "  ${GREEN}✓${NC} Fast Capture Engine - Built successfully"
elif [ "$CAPTURE_SUCCESS" = false ]; then
    echo -e "  ${RED}✗${NC} Fast Capture Engine - Build failed"
elif [ "$CAPTURE_SUCCESS" = skipped ]; then
    echo -e "  ${YELLOW}○${NC} Fast Capture Engine - Skipped"
fi

# Vision ML status
if [ "$VISION_ML_SUCCESS" = true ]; then
    echo -e "  ${GREEN}✓${NC} Vision ML Router - C++ acceleration enabled"
elif [ "$VISION_ML_SUCCESS" = false ]; then
    echo -e "  ${YELLOW}△${NC} Vision ML Router - Using Python fallback"
elif [ "$VISION_ML_SUCCESS" = skipped ]; then
    echo -e "  ${YELLOW}○${NC} Vision ML Router - Skipped"
fi

# Show usage instructions if anything was built
if [ "$CAPTURE_SUCCESS" = true ] || [ "$VISION_ML_SUCCESS" = true ]; then
    echo -e "\nTo use the extensions:"
    if [ "$CAPTURE_SUCCESS" = true ]; then
        echo -e "  ${YELLOW}Fast Capture:${NC}"
        echo -e "    from backend.native_extensions.fast_capture_wrapper import FastCaptureEngine"
    fi
    if [ "$VISION_ML_SUCCESS" = true ]; then
        echo -e "  ${YELLOW}Vision ML (C++):${NC}"
        echo -e "    import vision_ml_router  # C++ module"
    fi
    echo -e "  ${YELLOW}Hybrid Vision:${NC}"
    echo -e "    from backend.voice.hybrid_vision_router import HybridVisionRouter"
fi

echo -e "\nBuild Options:"
echo -e "  ${YELLOW}./build.sh${NC}          - Build all extensions"
echo -e "  ${YELLOW}./build.sh capture${NC}  - Build Fast Capture only"
echo -e "  ${YELLOW}./build.sh vision${NC}   - Build Vision ML only"
echo -e "  ${YELLOW}./build.sh clean${NC}    - Clean all builds"
echo -e "  ${YELLOW}./build.sh test${NC}     - Build all and run tests"

# Run tests if requested and builds succeeded
if [ "$RUN_TESTS" = true ]; then
    echo -e "\n${YELLOW}Running tests...${NC}"
    echo "================================"
    
    if [ "$CAPTURE_SUCCESS" = true ]; then
        echo -e "\n${YELLOW}Testing Fast Capture performance...${NC}"
        cd ../vision
        python3 test_enhanced_vision.py || echo -e "${YELLOW}Fast Capture test failed${NC}"
        cd ../native_extensions
    fi
    
    if [ "$VISION_ML_SUCCESS" = true ]; then
        echo -e "\n${YELLOW}Testing Vision ML Router...${NC}"
        python3 -c "
import vision_ml_router
test_commands = [
    'describe what is on my screen',
    'analyze the current window',
    'check for notifications'
]
print('Testing Vision ML analysis:')
for cmd in test_commands:
    score, action = vision_ml_router.analyze(cmd)
    print(f'  \"{cmd}\" -> {action} (score: {score:.2f})')
" || echo -e "${YELLOW}Vision ML test failed${NC}"
    fi
fi