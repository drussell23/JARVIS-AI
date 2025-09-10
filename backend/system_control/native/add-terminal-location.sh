#!/bin/bash

echo "🎯 Adding Terminal to Location Services"
echo "====================================="
echo ""

# Method 1: Use tccutil with correct syntax
echo "Method 1: Using tccutil to add Terminal..."
echo "This requires administrator privileges:"
echo ""

# Try to add Terminal to location services database
sudo sqlite3 /Library/Application\ Support/com.apple.TCC/TCC.db << 'EOF' 2>/dev/null || true
INSERT OR REPLACE INTO access (service, client, client_type, auth_value, auth_reason, auth_version) 
VALUES ('kTCCServiceLocationServices', 'com.apple.Terminal', 0, 2, 0, 1);
.quit
EOF

# Method 2: Create a location request from Terminal's bundle ID
echo ""
echo "Method 2: Creating location request with Terminal's bundle ID..."

# Create a tool that explicitly uses Terminal's bundle identifier
cat > terminal-location-request.m << 'EOF'
#import <Foundation/Foundation.h>
#import <CoreLocation/CoreLocation.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        NSLog(@"Requesting location for Terminal...");
        
        // Create location manager
        CLLocationManager *locationManager = [[CLLocationManager alloc] init];
        
        // Start location updates
        [locationManager startUpdatingLocation];
        
        // Try to get current location
        CLLocation *location = locationManager.location;
        if (location) {
            NSLog(@"Current location: %@", location);
        }
        
        // Keep running for a moment
        [[NSRunLoop currentRunLoop] runUntilDate:[NSDate dateWithTimeIntervalSinceNow:2.0]];
        
        NSLog(@"Done. Check Location Services for Terminal.");
    }
    return 0;
}
EOF

# Compile with Objective-C
echo "Compiling Objective-C version..."
clang -framework Foundation -framework CoreLocation terminal-location-request.m -o terminal-location-request

# Run it
echo "Running location request..."
./terminal-location-request

# Method 3: Direct manual instruction
echo ""
echo "====================================="
echo "📍 MANUAL METHOD (Most Reliable):"
echo "====================================="
echo ""
echo "Since Terminal won't appear automatically, here's what to do:"
echo ""
echo "1. Open System Preferences > Privacy & Security > Location Services"
echo ""
echo "2. Click the LOCK icon at the bottom left and enter your password"
echo ""
echo "3. Click the '+' (plus) button at the bottom of the app list"
echo ""
echo "4. In the file browser that opens:"
echo "   - Press Cmd+Shift+G to go to a specific folder"
echo "   - Enter: /System/Applications/Utilities/"
echo "   - Select 'Terminal.app' and click 'Open'"
echo ""
echo "5. Terminal will now appear in the list - check the box to enable it"
echo ""
echo "6. Click the lock again to save changes"
echo ""
echo "====================================="
echo ""
echo "Opening Location Services now..."
open "x-apple.systempreferences:com.apple.preference.security?Privacy_LocationServices"
echo ""
echo "Follow the manual steps above to add Terminal!"