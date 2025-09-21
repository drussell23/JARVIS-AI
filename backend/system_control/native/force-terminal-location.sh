#!/bin/bash

echo "🚀 Force Terminal to appear in Location Services"
echo "=============================================="
echo ""

# Step 1: Reset location services for Terminal (requires admin)
echo "Step 1: Resetting location services..."
echo "You may be prompted for your admin password:"
sudo tccutil reset LocationServices com.apple.Terminal 2>/dev/null || true
sudo tccutil reset Location com.apple.Terminal 2>/dev/null || true

# Step 2: Create a Location request that Terminal will be associated with
echo ""
echo "Step 2: Creating location request through Terminal..."

# Create a Swift script that explicitly runs in Terminal context
cat > force-location.swift << 'EOF'
import Foundation
import CoreLocation

print("Forcing Terminal to request location access...")

// Create location manager
let manager = CLLocationManager()

// Set delegate to nil to avoid complexity
manager.delegate = nil

// Multiple attempts to trigger location permission
print("Triggering location services...")
manager.startUpdatingLocation()
manager.requestLocation()

// Try to access location directly
if let location = manager.location {
    print("Current location: \(location)")
} else {
    print("No cached location available")
}

// Keep trying for a few seconds
for i in 1...3 {
    print("Attempt \(i)...")
    manager.startUpdatingLocation()
    Thread.sleep(forTimeInterval: 1.0)
}

print("\nDone! Check System Preferences > Security & Privacy > Privacy > Location Services")
print("Terminal should now appear in the list.")
EOF

# Compile and run it
swiftc force-location.swift -o force-location -framework CoreLocation
./force-location

# Step 3: Alternative method using osascript
echo ""
echo "Step 3: Using AppleScript to trigger location..."
osascript -e 'tell application "Terminal" to do script "echo \"Requesting location...\"; exit"'

# Step 4: Open Location Services preferences
echo ""
echo "Step 4: Opening Location Services preferences..."
open "x-apple.systempreferences:com.apple.preference.security?Privacy_LocationServices"

echo ""
echo "=============================================="
echo "✅ WHAT TO DO NOW:"
echo "=============================================="
echo ""
echo "1. Location Services preferences should now be open"
echo ""
echo "2. Look for 'Terminal' in the list"
echo "   - It may be at the bottom of the list"
echo "   - You might need to scroll down"
echo ""
echo "3. If Terminal appears:"
echo "   - Click the checkbox next to Terminal to enable it"
echo "   - You may need to click the lock icon first and enter your password"
echo ""
echo "4. If Terminal STILL doesn't appear:"
echo "   - Close System Preferences"
echo "   - Open a BRAND NEW Terminal window (Cmd+N)"
echo "   - Run this command again: ./force-terminal-location.sh"
echo ""
echo "5. Last resort - Manual method:"
echo "   - In System Preferences > Location Services"
echo "   - Click the '+' button at the bottom"
echo "   - Navigate to /System/Applications/Utilities/Terminal.app"
echo "   - Add it manually"
echo ""
echo "=============================================="