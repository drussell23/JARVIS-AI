#!/usr/bin/env python3
"""
Find the exact coordinates of Control Center icon
"""

import pyautogui
import time

print("\n" + "="*60)
print("🎯 Find Control Center Icon Coordinates")
print("="*60)

print("\n📍 Instructions:")
print("   1. Move your mouse over the Control Center icon")
print("   2. Wait for the coordinates to stabilize")
print("   3. Press Ctrl+C when done")
print()

try:
    while True:
        x, y = pyautogui.position()
        print(f"\r   Mouse position: ({x}, {y})   ", end='', flush=True)
        time.sleep(0.1)
except KeyboardInterrupt:
    x, y = pyautogui.position()
    print(f"\n\n✅ Final coordinates: ({x}, {y})")
    print("\nUse these coordinates for Control Center!")
    print("="*60)