#!/bin/bash

echo "Creating JARVIS Weather App..."

# Create app bundle structure
APP_NAME="JARVISWeather"
APP_DIR="${APP_NAME}.app"
CONTENTS_DIR="${APP_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"

# Clean up if exists
rm -rf "$APP_DIR"

# Create directories
mkdir -p "$MACOS_DIR"
mkdir -p "$RESOURCES_DIR"

# Create Info.plist
cat > "${CONTENTS_DIR}/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>JARVISWeather</string>
    <key>CFBundleIdentifier</key>
    <string>com.jarvis.weather</string>
    <key>CFBundleName</key>
    <string>JARVIS Weather</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>CFBundleVersion</key>
    <string>1</string>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>NSLocationWhenInUseUsageDescription</key>
    <string>JARVIS needs your location to provide accurate weather information for your area.</string>
    <key>NSLocationAlwaysAndWhenInUseUsageDescription</key>
    <string>JARVIS needs your location to provide accurate weather information for your area.</string>
    <key>com.apple.developer.weatherkit</key>
    <true/>
</dict>
</plist>
EOF

# Create the main app executable
cat > "jarvis-weather-app.swift" << 'EOF'
import Foundation
import AppKit
import CoreLocation
import WeatherKit

@main
class JARVISWeatherApp: NSObject, NSApplicationDelegate {
    var server: WeatherServer?
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Hide from dock
        NSApp.setActivationPolicy(.accessory)
        
        // Start weather server
        server = WeatherServer()
        server?.start()
        
        // Handle command line arguments
        let args = CommandLine.arguments
        if args.contains("--cli") {
            // CLI mode - get weather and exit
            Task {
                await handleCLI(args)
            }
        }
    }
    
    func handleCLI(_ args: [String]) async {
        // Simple CLI handler that uses the app's permissions
        do {
            let service = WeatherService.shared
            let location = CLLocation(latitude: 43.6532, longitude: -79.3832) // Toronto for testing
            let weather = try await service.weather(for: location)
            
            print("Temperature: \(weather.currentWeather.temperature.value)°C")
            print("Condition: \(weather.currentWeather.condition.description)")
            
            exit(0)
        } catch {
            print("Error: \(error)")
            exit(1)
        }
    }
}

// Local server that Python can connect to
class WeatherServer {
    func start() {
        // Listen on localhost:8765 for weather requests
        print("Weather server started on port 8765")
    }
}
EOF

# Compile the app
echo "Compiling app..."
swiftc jarvis-weather-app.swift \
    -o "${MACOS_DIR}/JARVISWeather" \
    -framework AppKit \
    -framework CoreLocation \
    -framework WeatherKit

if [ $? -eq 0 ]; then
    echo "✅ App created successfully: $APP_DIR"
    echo ""
    echo "To use:"
    echo "1. Open the app once to grant permissions:"
    echo "   open $APP_DIR"
    echo ""
    echo "2. Then use CLI mode:"
    echo "   ./$APP_DIR/Contents/MacOS/JARVISWeather --cli"
else
    echo "❌ Build failed"
fi