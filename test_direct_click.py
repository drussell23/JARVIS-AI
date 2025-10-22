#!/usr/bin/env python3
"""
Direct test of the 3-step click coordinates
"""

import pyautogui
import time

print("\n" + "="*60)
print("🎯 DIRECT COORDINATE TEST")
print("="*60)

print("\n📍 Clicking sequence in 3 seconds...")
print("  1. Control Center: (1236, 12)")
print("  2. Screen Mirroring: (1396, 177)")
print("  3. Living Room TV: (1223, 115)")

# Safety settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1

print("\n⏳ Starting in 3...")
time.sleep(1)
print("   2...")
time.sleep(1)
print("   1...")
time.sleep(1)

try:
    # Step 1: Control Center
    print("\n📍 Clicking Control Center (1236, 12)")
    pyautogui.moveTo(1236, 12, duration=0.2)
    pyautogui.click()
    time.sleep(0.8)  # Wait for menu

    # Step 2: Screen Mirroring
    print("📍 Clicking Screen Mirroring (1396, 177)")
    pyautogui.moveTo(1396, 177, duration=0.2)
    pyautogui.click()
    time.sleep(0.8)  # Wait for submenu

    # Step 3: Living Room TV
    print("📍 Clicking Living Room TV (1223, 115)")
    pyautogui.moveTo(1223, 115, duration=0.2)
    pyautogui.click()

    print("\n✅ Sequence complete!")

except Exception as e:
    print(f"\n❌ Error: {e}")
    # Clean up
    print("🧹 Pressing ESC to close menus...")
    pyautogui.press('escape')
    time.sleep(0.2)
    pyautogui.press('escape')

print("="*60)