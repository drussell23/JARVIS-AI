#!/usr/bin/env python3
"""Simple test to verify weather functionality"""

import subprocess
import time

print("🌤️ Simple Weather Test\n")

# Step 1: Open Weather app
print("1. Opening Weather app...")
subprocess.run(['open', '-a', 'Weather'], check=False)
time.sleep(2)

# Step 2: Navigate to My Location
print("2. Navigating to My Location...")
subprocess.run(['osascript', '-e', '''
tell application "System Events"
    key code 126
    delay 0.2
    key code 126
    delay 0.2
    key code 125
    delay 0.2
    key code 36
end tell
'''])

print("3. Weather app should now show My Location")
print("\n✅ If the Weather app is showing your local weather, the system is working!")
print("   JARVIS can now analyze this screen to tell you the weather.")