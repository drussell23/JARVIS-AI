# JARVIS Voice Unlock - Objective-C Build Configuration
# =====================================================

# Compiler and flags
CC = clang
OBJC = clang
CFLAGS = -fobjc-arc -Wall -Wextra -O2
FRAMEWORKS = -framework Foundation -framework AVFoundation -framework Security \
             -framework LocalAuthentication -framework IOKit -framework Carbon \
             -framework CoreGraphics -framework AudioToolbox -framework Accelerate \
             -framework AppKit

# Directories
SRC_DIR = .
BUILD_DIR = build
BIN_DIR = bin
DAEMON_DIR = daemon
FRAMEWORK_DIR = framework
BRIDGE_DIR = bridge
SECURITY_DIR = security
SERVER_DIR = server

# Source files
DAEMON_SOURCES = $(DAEMON_DIR)/JARVISVoiceUnlockDaemon.m
FRAMEWORK_SOURCES = $(FRAMEWORK_DIR)/JARVISVoiceAuthenticator.m \
                    $(FRAMEWORK_DIR)/JARVISVoiceMonitor.m
BRIDGE_SOURCES = $(BRIDGE_DIR)/JARVISWebSocketBridge.m \
                 $(BRIDGE_DIR)/JARVISPythonBridge.m
SECURITY_SOURCES = $(SECURITY_DIR)/JARVISScreenUnlockManager.m \
                   $(SECURITY_DIR)/JARVISPermissionManager.m
SERVER_SOURCES = $(SERVER_DIR)/JARVISWebSocketServer.m

ALL_SOURCES = $(DAEMON_SOURCES) $(FRAMEWORK_SOURCES) $(BRIDGE_SOURCES) $(SECURITY_SOURCES) $(SERVER_SOURCES)

# Object files
DAEMON_OBJECTS = $(patsubst $(DAEMON_DIR)/%.m,$(BUILD_DIR)/%.o,$(DAEMON_SOURCES))
FRAMEWORK_OBJECTS = $(patsubst $(FRAMEWORK_DIR)/%.m,$(BUILD_DIR)/%.o,$(FRAMEWORK_SOURCES))
BRIDGE_OBJECTS = $(patsubst $(BRIDGE_DIR)/%.m,$(BUILD_DIR)/%.o,$(BRIDGE_SOURCES))
SECURITY_OBJECTS = $(patsubst $(SECURITY_DIR)/%.m,$(BUILD_DIR)/%.o,$(SECURITY_SOURCES))
SERVER_OBJECTS = $(patsubst $(SERVER_DIR)/%.m,$(BUILD_DIR)/%.o,$(SERVER_SOURCES))

ALL_OBJECTS = $(DAEMON_OBJECTS) $(FRAMEWORK_OBJECTS) $(BRIDGE_OBJECTS) $(SECURITY_OBJECTS) $(SERVER_OBJECTS)

# Headers
HEADERS = $(wildcard $(DAEMON_DIR)/*.h) $(wildcard $(FRAMEWORK_DIR)/*.h) \
          $(wildcard $(BRIDGE_DIR)/*.h) $(wildcard $(SECURITY_DIR)/*.h) \
          $(wildcard $(SERVER_DIR)/*.h)

# Target executables
DAEMON_BINARY = $(BIN_DIR)/JARVISVoiceUnlockDaemon
TEST_BINARY = $(BIN_DIR)/JARVISVoiceUnlockTest

# Include paths
INCLUDES = -I$(DAEMON_DIR) -I$(FRAMEWORK_DIR) -I$(BRIDGE_DIR) -I$(SECURITY_DIR) -I$(SERVER_DIR)

# WebSocket support (using native NSURLSession)
# No external dependencies needed
SOCKETROCKET_FLAG =

# Default target
all: directories daemon

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build daemon executable
daemon: $(DAEMON_BINARY)

$(DAEMON_BINARY): $(ALL_OBJECTS) $(BUILD_DIR)/main.o
	@echo "Linking daemon..."
	$(OBJC) $(CFLAGS) $(FRAMEWORKS) $(SOCKETROCKET_FLAG) -o $@ $^
	@echo "Daemon built successfully: $@"

# Build test executable
test: $(TEST_BINARY)

$(TEST_BINARY): $(ALL_OBJECTS) $(BUILD_DIR)/test_main.o
	@echo "Linking test binary..."
	$(OBJC) $(CFLAGS) $(FRAMEWORKS) $(SOCKETROCKET_FLAG) -o $@ $^
	@echo "Test binary built successfully: $@"

# Compile object files
$(BUILD_DIR)/%.o: $(DAEMON_DIR)/%.m $(HEADERS)
	@echo "Compiling $<..."
	$(OBJC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(FRAMEWORK_DIR)/%.m $(HEADERS)
	@echo "Compiling $<..."
	$(OBJC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(BRIDGE_DIR)/%.m $(HEADERS)
	@echo "Compiling $<..."
	$(OBJC) $(CFLAGS) $(INCLUDES) $(SOCKETROCKET_FLAG) -c $< -o $@

$(BUILD_DIR)/%.o: $(SECURITY_DIR)/%.m $(HEADERS)
	@echo "Compiling $<..."
	$(OBJC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SERVER_DIR)/%.m $(HEADERS)
	@echo "Compiling $<..."
	$(OBJC) $(CFLAGS) $(INCLUDES) -c $< -o $@

# Main entry point
$(BUILD_DIR)/main.o: main.m $(HEADERS)
	@echo "Compiling main..."
	$(OBJC) $(CFLAGS) $(INCLUDES) -c main.m -o $(BUILD_DIR)/main.o

# Test main
test_main.o: test_main.m $(HEADERS)
	@echo "Compiling test main..."
	$(OBJC) $(CFLAGS) $(INCLUDES) -c test_main.m -o $(BUILD_DIR)/test_main.o

# Install daemon
install: daemon
	@echo "Installing JARVIS Voice Unlock Daemon..."
	@sudo cp $(DAEMON_BINARY) /usr/local/bin/
	@sudo cp com.jarvis.voiceunlock.plist /Library/LaunchDaemons/
	@sudo chown root:wheel /Library/LaunchDaemons/com.jarvis.voiceunlock.plist
	@sudo chmod 644 /Library/LaunchDaemons/com.jarvis.voiceunlock.plist
	@echo "Installation complete. Load daemon with:"
	@echo "  sudo launchctl load /Library/LaunchDaemons/com.jarvis.voiceunlock.plist"

# Uninstall daemon
uninstall:
	@echo "Uninstalling JARVIS Voice Unlock Daemon..."
	@sudo launchctl unload /Library/LaunchDaemons/com.jarvis.voiceunlock.plist 2>/dev/null || true
	@sudo rm -f /Library/LaunchDaemons/com.jarvis.voiceunlock.plist
	@sudo rm -f /usr/local/bin/JARVISVoiceUnlockDaemon
	@echo "Uninstallation complete."

# Code signing (requires valid Developer ID)
codesign: daemon
	@echo "Code signing daemon..."
	codesign -s "Developer ID Application: Your Name" --options runtime \
	         --entitlements entitlements.plist $(DAEMON_BINARY)

# Create installer package
package: daemon
	@echo "Creating installer package..."
	@mkdir -p package/usr/local/bin
	@mkdir -p package/Library/LaunchDaemons
	@cp $(DAEMON_BINARY) package/usr/local/bin/
	@cp com.jarvis.voiceunlock.plist package/Library/LaunchDaemons/
	pkgbuild --root package --identifier com.jarvis.voiceunlock \
	         --version 1.0 --install-location / JARVISVoiceUnlock.pkg
	@rm -rf package

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -rf $(BIN_DIR)
	@rm -f *.pkg

# Clean everything including installed files
distclean: clean uninstall
	@echo "Cleaned all files."

# Run tests
run-test: test
	@echo "Running tests..."
	@$(TEST_BINARY)

# Check dependencies
check-deps:
	@echo "Checking dependencies..."
	@which clang > /dev/null || (echo "Error: clang not found" && exit 1)
	@test -d "$(SOCKETROCKET_PATH)" || echo "Warning: SocketRocket.framework not found at $(SOCKETROCKET_PATH)"
	@echo "Dependencies check complete."

# Development helpers
format:
	@echo "Formatting code..."
	@find . -name "*.m" -o -name "*.h" | xargs clang-format -i

analyze:
	@echo "Running static analyzer..."
	$(OBJC) $(CFLAGS) $(INCLUDES) --analyze $(ALL_SOURCES)

# Help
help:
	@echo "JARVIS Voice Unlock - Build System"
	@echo "================================="
	@echo ""
	@echo "Available targets:"
	@echo "  make all       - Build the daemon (default)"
	@echo "  make daemon    - Build the daemon executable"
	@echo "  make test      - Build the test executable"
	@echo "  make install   - Install daemon and launch configuration"
	@echo "  make uninstall - Remove daemon and launch configuration"
	@echo "  make codesign  - Code sign the daemon (requires Developer ID)"
	@echo "  make package   - Create installer package"
	@echo "  make clean     - Remove build artifacts"
	@echo "  make distclean - Clean everything including installed files"
	@echo "  make run-test  - Build and run tests"
	@echo "  make check-deps- Check build dependencies"
	@echo "  make format    - Format source code"
	@echo "  make analyze   - Run static analyzer"
	@echo "  make help      - Show this help message"

.PHONY: all directories daemon test install uninstall codesign package \
        clean distclean run-test check-deps format analyze help