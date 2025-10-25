#!/usr/bin/env python3
"""
Test the exact flow JARVIS uses to see if coordinates get doubled
"""
import sys
import os
import time
import pyautogui

sys.path.insert(0, os.path.dirname(__file__))

print("\n" + "="*80)
print("TESTING EXACT JARVIS FLOW")
print("="*80 + "\n")

# Test direct PyAutoGUI first
print("1. Testing direct PyAutoGUI...")
pyautogui.moveTo(1236, 12, duration=0.5)
x1, y1 = pyautogui.position()
print(f"   Direct: Sent (1236, 12) → Got ({x1}, {y1})")

time.sleep(1)

# Now test through the simple clicker
print("\n2. Testing through control_center_clicker_simple...")
from backend.display.control_center_clicker_simple import get_control_center_clicker

clicker = get_control_center_clicker()
print(f"   Clicker coords: ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y})")

# Just test the movement, not clicking
pyautogui.moveTo(clicker.CONTROL_CENTER_X, clicker.CONTROL_CENTER_Y, duration=0.5)
x2, y2 = pyautogui.position()
print(f"   Via clicker: Sent ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y}) → Got ({x2}, {y2})")

time.sleep(1)

# Now test the full open_control_center method
print("\n3. Testing open_control_center method...")
result = clicker.open_control_center()
x3, y3 = pyautogui.position()
print(f"   After method: Mouse at ({x3}, {y3})")
print(f"   Method result: {result.get('success')}")

print("\n" + "="*80)
print("RESULTS:")
if x1 == 1236 and y1 == 12:
    print("✅ Direct PyAutoGUI works correctly")
else:
    print(f"❌ Direct PyAutoGUI failed: ({x1}, {y1})")

if x2 == clicker.CONTROL_CENTER_X and y2 == clicker.CONTROL_CENTER_Y:
    print("✅ Clicker coordinates work correctly")
else:
    print(f"❌ Clicker coordinates failed: ({x2}, {y2})")

if x3 == clicker.CONTROL_CENTER_X and y3 == clicker.CONTROL_CENTER_Y:
    print("✅ Method works correctly")
else:
    print(f"❌ Method failed: ({x3}, {y3})")

print("="*80 + "\n")