#!/bin/bash

echo "=== JARVIS Weather Location Permission Setup ==="
echo ""
echo "To use real-time location for weather, Terminal needs location permission."
echo ""
echo "Steps to enable:"
echo "1. Open System Preferences"
echo "2. Go to Security & Privacy > Privacy > Location Services"
echo "3. Click the lock to make changes"
echo "4. Find 'Terminal' in the list (or add it with +)"
echo "5. Check the box next to Terminal"
echo ""
echo "Alternative: Use city-based weather instead:"
echo "  ./jarvis-weather city Toronto"
echo ""
echo "Opening System Preferences now..."

# Open location services preferences
open "x-apple.systempreferences:com.apple.preference.security?Privacy_LocationServices"

echo ""
echo "After granting permission, test with:"
echo "  ./jarvis-weather current --pretty"