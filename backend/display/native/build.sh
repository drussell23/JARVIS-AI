#!/bin/bash
###############################################################################
# Native AirPlay Bridge Build Script
# ===================================
#
# Compiles the Swift native bridge with optimizations.
#
# Usage:
#   ./build.sh          # Build optimized binary
#   ./build.sh clean    # Clean build artifacts
#   ./build.sh test     # Build and run tests
#   ./build.sh install  # Build and install to PATH
#
# Author: Derek Russell
# Date: 2025-10-15
###############################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWIFT_FILE="$SCRIPT_DIR/AirPlayBridge.swift"
BINARY_NAME="AirPlayBridge"
BINARY_PATH="$SCRIPT_DIR/$BINARY_NAME"
BUILD_CACHE="$SCRIPT_DIR/.build_cache"
CONFIG_FILE="$SCRIPT_DIR/../../config/airplay_config.json"

# Functions
print_header() {
    echo -e "${BLUE}$1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

check_requirements() {
    print_header "Checking requirements..."
    
    # Check for Swift compiler
    if ! command -v swiftc &> /dev/null; then
        print_error "Swift compiler not found"
        echo "Please install Xcode or Swift toolchain"
        exit 1
    fi
    
    SWIFT_VERSION=$(swiftc --version | head -1)
    print_success "Swift found: $SWIFT_VERSION"
    
    # Check for config file
    if [ ! -f "$CONFIG_FILE" ]; then
        print_warning "Config file not found: $CONFIG_FILE"
        echo "Using default configuration..."
    fi
}

clean_build() {
    print_header "Cleaning build artifacts..."
    
    rm -rf "$BUILD_CACHE"
    rm -f "$BINARY_PATH"
    
    print_success "Clean complete"
}

build_bridge() {
    print_header "Building Native AirPlay Bridge..."
    
    # Create build cache directory
    mkdir -p "$BUILD_CACHE"
    
    # Compile with optimizations
    print_header "Compiling Swift code..."
    
    swiftc "$SWIFT_FILE" \
        -o "$BINARY_PATH" \
        -framework Foundation \
        -framework CoreGraphics \
        -framework ApplicationServices \
        -framework IOKit \
        -framework Cocoa \
        -parse-as-library \
        -O \
        -whole-module-optimization 2>&1 | tee "$BUILD_CACHE/build.log"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        # Save source hash
        SOURCE_HASH=$(shasum -a 256 "$SWIFT_FILE" | awk '{print $1}')
        echo "$SOURCE_HASH" > "$BUILD_CACHE/source_hash.txt"
        
        # Get binary size
        BINARY_SIZE=$(ls -lh "$BINARY_PATH" | awk '{print $5}')
        
        print_success "Compilation successful"
        echo "  Binary: $BINARY_PATH"
        echo "  Size: $BINARY_SIZE"
        
        # Make executable
        chmod +x "$BINARY_PATH"
        
        return 0
    else
        print_error "Compilation failed"
        echo "See $BUILD_CACHE/build.log for details"
        return 1
    fi
}

test_bridge() {
    print_header "Testing Native AirPlay Bridge..."
    
    if [ ! -f "$BINARY_PATH" ]; then
        print_error "Binary not found. Run './build.sh' first"
        exit 1
    fi
    
    # Test 1: Discover displays
    print_header "Test 1: Discovering displays..."
    if "$BINARY_PATH" discover "$CONFIG_FILE" 2>/dev/null | python3 -m json.tool; then
        print_success "Discovery test passed"
    else
        print_warning "Discovery test failed (may be normal if no displays available)"
    fi
    
    echo ""
    
    # Test 2: Check accessibility permissions
    print_header "Test 2: Checking accessibility permissions..."
    PERMISSIONS=$(osascript -e 'tell application "System Events" to get UI elements enabled' 2>/dev/null || echo "false")
    
    if [ "$PERMISSIONS" = "true" ]; then
        print_success "Accessibility permissions granted"
    else
        print_warning "Accessibility permissions not granted"
        echo "  To grant: System Settings → Privacy & Security → Accessibility"
        echo "  Add: Terminal (or your Python interpreter)"
    fi
    
    echo ""
    print_success "Tests complete"
}

install_bridge() {
    print_header "Installing Native AirPlay Bridge..."
    
    if [ ! -f "$BINARY_PATH" ]; then
        print_error "Binary not found. Run './build.sh' first"
        exit 1
    fi
    
    INSTALL_DIR="$HOME/.local/bin"
    mkdir -p "$INSTALL_DIR"
    
    cp "$BINARY_PATH" "$INSTALL_DIR/"
    
    print_success "Installed to $INSTALL_DIR/$BINARY_NAME"
    
    # Check if in PATH
    if [[ ":$PATH:" != *":$INSTALL_DIR:"* ]]; then
        print_warning "Add to PATH: export PATH=\"\$HOME/.local/bin:\$PATH\""
    fi
}

show_usage() {
    cat << EOF
Native AirPlay Bridge Build Script

Usage:
    ./build.sh          Build optimized binary
    ./build.sh clean    Clean build artifacts
    ./build.sh test     Build and run tests
    ./build.sh install  Build and install to ~/.local/bin
    ./build.sh help     Show this help

Examples:
    # Build the bridge
    ./build.sh

    # Clean and rebuild
    ./build.sh clean && ./build.sh

    # Build and test
    ./build.sh test

EOF
}

# Main
main() {
    print_header "╔════════════════════════════════════════╗"
    print_header "║  Native AirPlay Bridge Build System   ║"
    print_header "╚════════════════════════════════════════╝"
    echo ""
    
    case "${1:-build}" in
        build)
            check_requirements
            build_bridge
            ;;
        clean)
            clean_build
            ;;
        test)
            check_requirements
            if [ ! -f "$BINARY_PATH" ]; then
                build_bridge
            fi
            test_bridge
            ;;
        install)
            check_requirements
            if [ ! -f "$BINARY_PATH" ]; then
                build_bridge
            fi
            install_bridge
            ;;
        help|--help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown command: $1"
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
