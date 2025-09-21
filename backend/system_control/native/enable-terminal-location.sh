#!/bin/bash

echo "🌍 Enabling Terminal Location Services"
echo "===================================="
echo ""

# Method 1: Create a Location Services entry for Terminal
echo "Method 1: Creating Location Services entry..."
echo ""

# Create a simple macOS app that will request location on behalf of Terminal
cat > terminal-location-helper.swift << 'EOF'
import Foundation
import CoreLocation
import AppKit

// Create a minimal macOS app that requests location permission
class TerminalLocationHelper: NSObject, NSApplicationDelegate, CLLocationManagerDelegate {
    let locationManager = CLLocationManager()
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        // Set the app to run in background
        NSApp.setActivationPolicy(.accessory)
        
        locationManager.delegate = self
        
        print("Requesting location permission for Terminal...")
        
        // This will trigger the system permission dialog
        locationManager.startUpdatingLocation()
        
        // Give it a moment then quit
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            NSApp.terminate(nil)
        }
    }
    
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        print("✅ Location permission granted!")
        NSApp.terminate(nil)
    }
    
    func locationManager(_ manager: CLLocationManager, didFailWithError error: Error) {
        print("Location error: \(error)")
    }
}

// Create and run the app
let app = NSApplication.shared
let delegate = TerminalLocationHelper()
app.delegate = delegate
app.run()
EOF

# Compile it
echo "Compiling helper..."
swiftc terminal-location-helper.swift -o terminal-location-helper -framework CoreLocation -framework AppKit

# Method 2: Use Python to trigger location request
echo ""
echo "Method 2: Using Python CoreLocation..."
python3 << 'EOF'
try:
    from CoreLocation import CLLocationManager
    print("Attempting to access location via Python...")
    manager = CLLocationManager.alloc().init()
    manager.startUpdatingLocation()
    print("Location request initiated")
except Exception as e:
    print(f"Python method failed: {e}")
EOF

echo ""
echo "Method 3: Direct system call..."
# Try using the location services directly
osascript << 'EOF'
try
    do shell script "CoreLocationCLI -once" with administrator privileges
on error
    display dialog "Please install CoreLocationCLI for better location support" buttons {"OK"}
end try
EOF

echo ""
echo "================================================================"
echo "IMPORTANT STEPS:"
echo "================================================================"
echo ""
echo "1. Run the helper app we just created:"
echo "   ./terminal-location-helper"
echo ""
echo "2. When the permission dialog appears:"
echo "   - Click 'Allow' to grant location access"
echo ""
echo "3. Then check System Settings > Privacy & Security > Location Services"
echo "   - Terminal should now appear in the list"
echo ""
echo "4. If Terminal still doesn't appear, try:"
echo "   a) Open a NEW Terminal window"
echo "   b) Run: ./terminal-location-helper"
echo "   c) Grant permission when prompted"
echo ""
echo "5. Alternative: Install CoreLocationCLI"
echo "   brew install corelocationcli"
echo "   Then run: CoreLocationCLI -once"
echo ""
echo "================================================================"