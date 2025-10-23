#!/usr/bin/env python3
"""
Test to identify where coordinate doubling occurs in PyAutoGUI
"""

import pyautogui
import time

print("=" * 80)
print("PyAutoGUI Coordinate Doubling Test")
print("=" * 80)

# Get screen info
screen_width, screen_height = pyautogui.size()
print(f"\nScreen size: {screen_width}x{screen_height}")

# Test coordinates
test_coords = [
    (1235, 10, "Control Center"),
    (1236, 12, "Control Center Alt"),
    (100, 100, "Safe zone"),
]

for x, y, name in test_coords:
    print(f"\n{'-' * 80}")
    print(f"Testing: {name}")
    print(f"Target coordinates: ({x}, {y})")

    # Test moveTo
    print(f"\n1. Testing moveTo({x}, {y})...")
    pyautogui.moveTo(x, y, duration=0.1)
    pos_after_move = pyautogui.position()
    print(f"   After moveTo: Mouse at ({pos_after_move.x}, {pos_after_move.y})")
    if pos_after_move.x != x or pos_after_move.y != y:
        print(f"   ❌ MISMATCH! Expected ({x}, {y})")
        if pos_after_move.x > x * 1.5:
            print(f"   🚨 DOUBLING DETECTED! {pos_after_move.x} ≈ {x} × 2")
    else:
        print(f"   ✅ Correct position")

    time.sleep(0.5)

    # Test dragTo
    print(f"\n2. Testing dragTo({x}, {y})...")
    # Move to a different position first
    pyautogui.moveTo(x - 50, y - 50, duration=0.1)
    time.sleep(0.1)

    pyautogui.dragTo(x, y, duration=0.1, button='left')
    pos_after_drag = pyautogui.position()
    print(f"   After dragTo: Mouse at ({pos_after_drag.x}, {pos_after_drag.y})")
    if pos_after_drag.x != x or pos_after_drag.y != y:
        print(f"   ❌ MISMATCH! Expected ({x}, {y})")
        if pos_after_drag.x > x * 1.5:
            print(f"   🚨 DOUBLING DETECTED! {pos_after_drag.x} ≈ {x} × 2")
    else:
        print(f"   ✅ Correct position")

    time.sleep(0.5)

print(f"\n{'=' * 80}")
print("Test complete!")
print("=" * 80)
