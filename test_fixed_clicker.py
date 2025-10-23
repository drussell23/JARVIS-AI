#!/usr/bin/env python3
"""
Test that the control_center_clicker now uses simple coordinates by default
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from backend.display.control_center_clicker import get_control_center_clicker

def main():
    print("\n" + "="*80)
    print("Testing Fixed Control Center Clicker")
    print("="*80 + "\n")
    
    # Get the clicker (should now default to use_adaptive=False)
    clicker = get_control_center_clicker()
    
    print(f"Clicker configuration:")
    print(f"  use_adaptive: {clicker.use_adaptive} (should be False)")
    
    if not clicker.use_adaptive:
        print(f"\n✅ Using SIMPLE hardcoded coordinates:")
        print(f"  Control Center:   ({clicker.CONTROL_CENTER_X}, {clicker.CONTROL_CENTER_Y})")
        print(f"  Screen Mirroring: ({clicker.SCREEN_MIRRORING_X}, {clicker.SCREEN_MIRRORING_Y})")
        print(f"  Living Room TV:   ({clicker.LIVING_ROOM_TV_X}, {clicker.LIVING_ROOM_TV_Y})")
        
        print("\n✅ FIXED: No more coordinate doubling issues!")
        print("  The clicker will now use (1236,12) not (2475,15)")
    else:
        print("\n❌ WARNING: Still using adaptive mode!")
        print("  This may cause coordinate doubling issues")
    
    print("\n" + "-"*80)
    print("Testing connection flow...")
    print("-"*80 + "\n")
    
    input("Press ENTER to test connection...")
    
    result = clicker.connect_to_living_room_tv()
    
    print("\nResult:")
    print(f"  Success: {result['success']}")
    print(f"  Method: {result.get('method', 'unknown')}")
    if result['success']:
        print(f"  Coordinates used:")
        print(f"    Control Center: {result.get('control_center_coords')}")
        print(f"    Screen Mirroring: {result.get('screen_mirroring_coords')}")
        print(f"    Living Room TV: {result.get('living_room_tv_coords')}")
    else:
        print(f"  Error: {result.get('message')}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
