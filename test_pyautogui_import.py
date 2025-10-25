#!/usr/bin/env python3
"""
Test what happens when we import things in the same order as JARVIS
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

print("\n" + "="*80)
print("TESTING IMPORT ORDER EFFECTS")
print("="*80 + "\n")

# Import in the same order that might happen in JARVIS
print("1. Importing backend modules...")
from backend.display.adaptive_control_center_clicker import AdaptiveControlCenterClicker
print("   - adaptive_control_center_clicker imported")

print("\n2. Testing PyAutoGUI after imports...")
import pyautogui

# Check if pyautogui has been modified
print(f"   pyautogui.moveTo: {pyautogui.moveTo}")
print(f"   pyautogui.click: {pyautogui.click}")

# Test actual movement
print("\n3. Testing coordinate (1236, 12)...")
pyautogui.moveTo(1236, 12, duration=0.5)
actual_x, actual_y = pyautogui.position()

print(f"   Sent: (1236, 12)")
print(f"   Got: ({actual_x}, {actual_y})")

if actual_x == 1236 and actual_y == 12:
    print("   ✅ CORRECT - No doubling")
else:
    print(f"   ❌ WRONG - Coordinates changed!")
    if actual_x > 2000:
        print(f"   ⚠️  Likely doubled! {actual_x} ≈ 1236 × 2")

print("\n" + "="*80 + "\n")