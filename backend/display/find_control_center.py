#!/usr/bin/env python3
"""
Interactive Control Center Finder
Move your mouse to the Control Center icon and press ENTER
"""

import pyautogui
import time

print("\n" + "="*70)
print("🎯 Interactive Control Center Position Finder")
print("="*70)
print("\n1. Move your mouse cursor to the CENTER of the Control Center icon")
print("   (the one circled in red in your screenshot)")
print("\n2. Once your mouse is over the icon, press ENTER in this terminal")
print("\nWaiting for you to position mouse and press ENTER...")

input()

# Get mouse position
x, y = pyautogui.position()

print(f"\n✅ Control Center icon position: ({x}, {y})")
print(f"\nScreen size: {pyautogui.size()}")

# Calculate offset from right edge
screen_width = pyautogui.size()[0]
offset_from_right = screen_width - x

print(f"Offset from right edge: {offset_from_right}px")

print(f"\n📝 Update the heuristic in vision_ui_navigator.py:")
print(f"   OLD: x = screen_width - 70")
print(f"   NEW: x = screen_width - {offset_from_right}")

print(f"\n🎯 Or use absolute position:")
print(f"   x = {x}")
print(f"   y = {y}")

print("\n" + "="*70)
