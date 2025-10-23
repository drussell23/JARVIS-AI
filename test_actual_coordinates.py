#!/usr/bin/env python3
"""
Test what coordinates are actually being used by the control_center_clicker
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

def test_control_center_clicker():
    """Test control_center_clicker.py"""
    print("\n" + "="*80)
    print("Testing control_center_clicker.py")
    print("="*80 + "\n")

    from backend.display.control_center_clicker import get_control_center_clicker

    clicker = get_control_center_clicker(use_adaptive=False)

    print(f"use_adaptive: {clicker.use_adaptive}")
    print(f"CONTROL_CENTER_X: {clicker.CONTROL_CENTER_X}")
    print(f"CONTROL_CENTER_Y: {clicker.CONTROL_CENTER_Y}")

    # Check if coordinates are correct
    if clicker.CONTROL_CENTER_X == 1236 and clicker.CONTROL_CENTER_Y == 12:
        print("✅ Coordinates are CORRECT")
    else:
        print(f"❌ Coordinates are WRONG!")
        print(f"   Expected: (1236, 12)")
        print(f"   Got: ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y})")

def test_control_center_clicker_simple():
    """Test control_center_clicker_simple.py"""
    print("\n" + "="*80)
    print("Testing control_center_clicker_simple.py")
    print("="*80 + "\n")

    from backend.display.control_center_clicker_simple import get_control_center_clicker

    clicker = get_control_center_clicker()

    print(f"CONTROL_CENTER_X: {clicker.CONTROL_CENTER_X}")
    print(f"CONTROL_CENTER_Y: {clicker.CONTROL_CENTER_Y}")

    # Check if coordinates are correct
    if clicker.CONTROL_CENTER_X == 1236 and clicker.CONTROL_CENTER_Y == 12:
        print("✅ Coordinates are CORRECT")
    else:
        print(f"❌ Coordinates are WRONG!")
        print(f"   Expected: (1236, 12)")
        print(f"   Got: ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y})")

def test_what_jarvis_uses():
    """Test what JARVIS actually imports"""
    print("\n" + "="*80)
    print("Testing what advanced_display_monitor imports")
    print("="*80 + "\n")

    # Check what's in the advanced_display_monitor
    with open("/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend/display/advanced_display_monitor.py", "r") as f:
        content = f.read()

    # Find the import line
    import_lines = [line for line in content.split("\n") if "control_center_clicker" in line and "import" in line]

    for line in import_lines[:5]:  # Show first 5 matching lines
        print(f"  {line.strip()}")

    # Check what coordinates would be used
    print("\n" + "-"*40)
    print("Checking what coordinates would be used:")
    print("-"*40 + "\n")

    # Find the line that imports the clicker
    if "control_center_clicker_simple import get_control_center_clicker" in content:
        print("✅ Imports from control_center_clicker_simple.py")
        from backend.display.control_center_clicker_simple import get_control_center_clicker
    else:
        print("❌ Does NOT import from control_center_clicker_simple.py")
        print("   Looking for other imports...")

    clicker = get_control_center_clicker()
    print(f"\nCoordinates that would be used:")
    print(f"  Control Center: ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y})")

    if clicker.CONTROL_CENTER_X == 1236 and clicker.CONTROL_CENTER_Y == 12:
        print("  ✅ These are CORRECT")
    else:
        print(f"  ❌ These are WRONG (should be 1236, 12)")

if __name__ == "__main__":
    test_control_center_clicker()
    test_control_center_clicker_simple()
    test_what_jarvis_uses()