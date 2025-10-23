#!/usr/bin/env python3
"""
Test PyAutoGUI directly to see if coordinates get doubled
"""
import pyautogui
import time
import sys
import os

# Get screen info
print("\n" + "="*80)
print("PYAUTOGUI SYSTEM INFO")
print("="*80 + "\n")

width, height = pyautogui.size()
print(f"Screen Size: {width} x {height}")

# Check for any DPI scaling
try:
    from AppKit import NSScreen
    scale = NSScreen.mainScreen().backingScaleFactor()
    print(f"macOS Backing Scale Factor: {scale}")
except:
    print("Could not get macOS backing scale factor")

# Check PyAutoGUI settings
print(f"\nPyAutoGUI Settings:")
print(f"  FAILSAFE: {pyautogui.FAILSAFE}")
print(f"  PAUSE: {pyautogui.PAUSE}")

# Test moving to exact coordinates
print("\n" + "="*80)
print("TESTING EXACT COORDINATE MOVEMENT")
print("="*80 + "\n")

test_coords = [
    (1236, 12, "Control Center"),
    (1393, 177, "Screen Mirroring"),
    (1221, 116, "Living Room TV")
]

for target_x, target_y, name in test_coords:
    print(f"\nTest: {name}")
    print(f"  Target: ({target_x}, {target_y})")

    # Move to position
    pyautogui.moveTo(target_x, target_y, duration=0.5)

    # Get actual position
    actual_x, actual_y = pyautogui.position()
    print(f"  Actual: ({actual_x}, {actual_y})")

    # Check if it matches
    if actual_x == target_x and actual_y == target_y:
        print(f"  ✅ CORRECT")
    else:
        print(f"  ❌ MISMATCH!")
        x_ratio = actual_x / target_x if target_x != 0 else 0
        y_ratio = actual_y / target_y if target_y != 0 else 0
        print(f"     X ratio: {x_ratio:.2f} (actual/target)")
        print(f"     Y ratio: {y_ratio:.2f} (actual/target)")

        if abs(x_ratio - 2.0) < 0.1:
            print(f"     ⚠️  X appears to be DOUBLED!")
        if abs(y_ratio - 2.0) < 0.1:
            print(f"     ⚠️  Y appears to be DOUBLED!")

    time.sleep(1)

print("\n" + "="*80)

# Now test if importing from JARVIS changes behavior
print("\nTesting after importing JARVIS modules...")
print("-"*80 + "\n")

sys.path.insert(0, os.path.dirname(__file__))
from backend.display.control_center_clicker import get_control_center_clicker

clicker = get_control_center_clicker(use_adaptive=False)

# Test again after import
print("Moving to Control Center coordinates from clicker...")
print(f"  Clicker coords: ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y})")

pyautogui.moveTo(clicker.CONTROL_CENTER_X, clicker.CONTROL_CENTER_Y, duration=0.5)
actual_x, actual_y = pyautogui.position()

print(f"  Actual position: ({actual_x}, {actual_y})")

if actual_x == clicker.CONTROL_CENTER_X and actual_y == clicker.CONTROL_CENTER_Y:
    print(f"  ✅ CORRECT after import")
else:
    print(f"  ❌ MISMATCH after import!")
    print(f"     Expected: ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y})")
    print(f"     Got: ({actual_x}, {actual_y})")

print("\n" + "="*80 + "\n")