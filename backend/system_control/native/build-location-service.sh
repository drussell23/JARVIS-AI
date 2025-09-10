#!/bin/bash

echo "🚀 Building JARVIS Location Service"
echo "=================================="

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
APP_NAME="JarvisLocationService"
BUNDLE_NAME="${APP_NAME}.app"
BUNDLE_ID="com.jarvis.location-service"

# Clean previous build
echo -e "${BLUE}Cleaning previous build...${NC}"
rm -rf "$BUNDLE_NAME"
rm -f "$APP_NAME"

# Create app bundle structure
echo -e "${BLUE}Creating app bundle structure...${NC}"
mkdir -p "${BUNDLE_NAME}/Contents/MacOS"
mkdir -p "${BUNDLE_NAME}/Contents/Resources"

# Create Info.plist
echo -e "${BLUE}Creating Info.plist...${NC}"
cat > "${BUNDLE_NAME}/Contents/Info.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>JarvisLocationService</string>
    <key>CFBundleIdentifier</key>
    <string>${BUNDLE_ID}</string>
    <key>CFBundleName</key>
    <string>JARVIS Location Service</string>
    <key>CFBundleDisplayName</key>
    <string>JARVIS Location Service</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleSignature</key>
    <string>????</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSUIElement</key>
    <true/>
    <key>NSLocationWhenInUseUsageDescription</key>
    <string>JARVIS needs your location to provide accurate weather information and location-based services. Your location data stays on your device and is only used to enhance your experience.</string>
    <key>NSLocationAlwaysAndWhenInUseUsageDescription</key>
    <string>JARVIS can provide better contextual assistance with continuous location access. Your location data is processed locally and never sent to external servers.</string>
    <key>NSLocationUsageDescription</key>
    <string>JARVIS uses your location to provide weather updates, contextual information, and location-based reminders.</string>
</dict>
</plist>
EOF

# Compile Swift code
echo -e "${BLUE}Compiling Swift code...${NC}"
if swiftc JarvisLocationService.swift \
    -o "${BUNDLE_NAME}/Contents/MacOS/JarvisLocationService" \
    -framework CoreLocation \
    -framework Foundation \
    -O; then
    echo -e "${GREEN}✅ Compilation successful${NC}"
else
    echo -e "${YELLOW}⚠️  Compilation failed${NC}"
    exit 1
fi

# Create standalone CLI version too
echo -e "${BLUE}Creating CLI version...${NC}"
cp "${BUNDLE_NAME}/Contents/MacOS/JarvisLocationService" "./jarvis-location"
chmod +x "./jarvis-location"

# Sign the app bundle
echo -e "${BLUE}Signing app bundle...${NC}"
if codesign --force --deep --sign - "$BUNDLE_NAME" 2>/dev/null; then
    echo -e "${GREEN}✅ App signed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Could not sign app (normal without developer certificate)${NC}"
fi

# Create Python wrapper
echo -e "${BLUE}Creating Python wrapper...${NC}"
cat > location_service.py << 'PYTHON_EOF'
"""
JARVIS Location Service Python Wrapper
Provides precise location data using macOS Core Location
"""

import json
import subprocess
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class LocationService:
    """Bridge to Swift Core Location service"""
    
    def __init__(self):
        self.service_dir = Path(__file__).parent
        self.app_bundle = self.service_dir / "JarvisLocationService.app"
        self.cli_binary = self.service_dir / "jarvis-location"
        self.cache_file = self.service_dir / ".location_cache.json"
        self.cache_duration = timedelta(minutes=5)
        
    def get_current_location(self) -> Optional[Dict]:
        """Get current location with caching"""
        # Check cache first
        cached = self._get_cached_location()
        if cached:
            logger.info(f"Using cached location: {cached.get('city', 'Unknown')}")
            return cached
            
        # Try to get fresh location
        location = self._request_location()
        
        if location and location.get('status') == 'success':
            self._save_to_cache(location)
            logger.info(f"Got fresh location: {location.get('city', 'Unknown')}")
            return location
        
        return None
        
    def _request_location(self) -> Optional[Dict]:
        """Request location from Swift service"""
        try:
            # Try app bundle first (has better permissions)
            if self.app_bundle.exists():
                executable = self.app_bundle / "Contents" / "MacOS" / "JarvisLocationService"
                cmd = [str(executable)]
            elif self.cli_binary.exists():
                cmd = [str(self.cli_binary)]
            else:
                logger.error("No location service binary found")
                return None
                
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and result.stdout:
                return json.loads(result.stdout)
            else:
                logger.error(f"Location service failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Location service timed out")
            return None
        except json.JSONDecodeError:
            logger.error("Invalid JSON from location service")
            return None
        except Exception as e:
            logger.error(f"Location service error: {e}")
            return None
            
    def _get_cached_location(self) -> Optional[Dict]:
        """Get location from cache if valid"""
        try:
            if not self.cache_file.exists():
                return None
                
            with open(self.cache_file, 'r') as f:
                cache = json.load(f)
                
            # Check if cache is still valid
            cached_time = datetime.fromisoformat(cache['timestamp'].replace('Z', '+00:00'))
            if datetime.now().astimezone() - cached_time < self.cache_duration:
                return cache
                
        except Exception:
            pass
            
        return None
        
    def _save_to_cache(self, location: Dict):
        """Save location to cache"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(location, f)
        except Exception as e:
            logger.error(f"Failed to save location cache: {e}")
            
    def get_coordinates(self) -> Optional[Tuple[float, float]]:
        """Get just coordinates (latitude, longitude)"""
        location = self.get_current_location()
        if location and location.get('status') == 'success':
            return (location['latitude'], location['longitude'])
        return None
        
    def get_city_info(self) -> Optional[Dict]:
        """Get city, region, country info"""
        location = self.get_current_location()
        if location and location.get('status') == 'success':
            return {
                'city': location.get('city'),
                'region': location.get('region'),
                'country': location.get('country'),
                'coordinates': (location['latitude'], location['longitude'])
            }
        return None


# Quick test when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    service = LocationService()
    location = service.get_current_location()
    
    if location:
        print(f"Location: {location.get('city', 'Unknown')}, {location.get('region', 'Unknown')}")
        print(f"Coordinates: {location['latitude']}, {location['longitude']}")
        print(f"Accuracy: {location['accuracy']}m")
    else:
        print("Failed to get location")
PYTHON_EOF

echo ""
echo -e "${GREEN}=================================="
echo "✅ Build Complete!"
echo "==================================${NC}"
echo ""
echo -e "${YELLOW}FIRST TIME SETUP:${NC}"
echo ""
echo "1. Run the app to trigger permission request:"
echo -e "   ${BLUE}open ${BUNDLE_NAME}${NC}"
echo ""
echo "2. macOS will show a permission dialog:"
echo "   'JARVIS Location Service would like to use your current location'"
echo "   Click 'Allow'"
echo ""
echo "3. Test the service:"
echo -e "   ${BLUE}./jarvis-location${NC}"
echo ""
echo "4. Test Python integration:"
echo -e "   ${BLUE}python location_service.py${NC}"
echo ""
echo -e "${GREEN}INTEGRATION:${NC}"
echo ""
echo "The service provides:"
echo "  • Precise GPS coordinates"
echo "  • City, region, country information"
echo "  • Accuracy measurements"
echo "  • 5-minute caching for efficiency"
echo "  • Proper error handling"
echo ""
echo "Your weather queries will now use YOUR exact location,"
echo "just like the macOS Weather app!"