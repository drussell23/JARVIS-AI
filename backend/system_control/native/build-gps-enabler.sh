#!/bin/bash

echo "🛠️ Building GPS Location Enabler"

# Clean
rm -rf GPSLocationEnabler.app

# Create app bundle
mkdir -p GPSLocationEnabler.app/Contents/MacOS
mkdir -p GPSLocationEnabler.app/Contents/Resources

# Create Info.plist
cat > GPSLocationEnabler.app/Contents/Info.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>GPSLocationEnabler</string>
    <key>CFBundleIdentifier</key>
    <string>com.jarvis.gps-location-enabler</string>
    <key>CFBundleName</key>
    <string>GPS Location Enabler</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>NSLocationWhenInUseUsageDescription</key>
    <string>JARVIS needs GPS location to provide accurate weather and location-based services in your exact location, not just city-level data.</string>
    <key>NSLocationAlwaysAndWhenInUseUsageDescription</key>
    <string>JARVIS can provide better contextual assistance with GPS location access for weather, navigation, and location reminders.</string>
</dict>
</plist>
EOF

# Compile
swiftc enable-gps-location.swift \
    -o GPSLocationEnabler.app/Contents/MacOS/GPSLocationEnabler \
    -framework CoreLocation \
    -framework AppKit

# Sign
codesign --force --sign - GPSLocationEnabler.app 2>/dev/null

echo "✅ Build complete!"
echo ""
echo "To enable GPS location:"
echo "1. Run: open GPSLocationEnabler.app"
echo "2. Click 'Enable GPS Location'"
echo "3. Allow location access when prompted"
echo "4. Click 'Test Location' to verify"