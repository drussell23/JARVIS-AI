#!/usr/bin/env python3
"""
Test the simple clicker based on working commit a7fd379
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from backend.display.control_center_clicker_simple import get_control_center_clicker

def main():
    print("\n" + "="*80)
    print("Testing Simple Control Center Clicker")
    print("Based on WORKING coordinates from commit a7fd379")
    print("="*80)
    
    clicker = get_control_center_clicker()
    
    print("\nCoordinates to be used:")
    print(f"  1. Control Center:   ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y})")
    print(f"  2. Screen Mirroring: ({clicker.SCREEN_MIRRORING_X}, {clicker.SCREEN_MIRRORING_Y})")
    print(f"  3. Living Room TV:   ({clicker.LIVING_ROOM_TV_X}, {clicker.LIVING_ROOM_TV_Y})")
    
    print("\nThis will perform 3 clicks in sequence.")
    print("Watch the mouse movement!\n")
    
    input("Press ENTER to start...")
    
    print("\n🎯 Executing connection flow...")
    result = clicker.connect_to_living_room_tv()
    
    print("\n" + "="*80)
    print("RESULT:")
    if result['success']:
        print("✅ SUCCESS!")
        print(f"  Method: {result['method']}")
        print(f"  Control Center: {result['control_center_coords']}")
        print(f"  Screen Mirroring: {result['screen_mirroring_coords']}")
        print(f"  Living Room TV: {result['living_room_tv_coords']}")
    else:
        print(f"❌ FAILED: {result['message']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
