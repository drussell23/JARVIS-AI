#!/usr/bin/env python3
"""
Debug script to inject into JARVIS to see what coordinates are being used
Add this to the control_center_clicker files to debug
"""

def debug_inject():
    """
    Add this function call to control_center_clicker.py or control_center_clicker_simple.py
    to see what coordinates are actually being used when JARVIS runs
    """
    import pyautogui
    import traceback
    import sys

    # Save original functions
    original_moveTo = pyautogui.moveTo
    original_click = pyautogui.click

    def debug_moveTo(x, y, duration=0.0, *args, **kwargs):
        """Wrapper to log all moveTo calls"""
        # Get the call stack
        stack = traceback.extract_stack()
        caller = stack[-2]  # The function that called moveTo

        # Log to a file
        with open("/tmp/jarvis_coordinate_debug.log", "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"PyAutoGUI.moveTo({x}, {y}, duration={duration})\n")
            f.write(f"Called from: {caller.filename}:{caller.lineno} in {caller.name}\n")
            f.write(f"Code: {caller.line}\n")

            # Check if coordinates are doubled
            if x > 2000:
                f.write(f"⚠️ WARNING: X coordinate {x} seems too large!\n")
                f.write(f"   Might be doubled from {x//2}\n")
            if y > 1000:
                f.write(f"⚠️ WARNING: Y coordinate {y} seems too large!\n")
                f.write(f"   Might be doubled from {y//2}\n")

        # Print to console too
        print(f"\n🎯 DEBUG: pyautogui.moveTo({x}, {y}, duration={duration})")
        print(f"   Called from: {caller.name} in {caller.filename}:{caller.lineno}")

        # Call original
        return original_moveTo(x, y, duration, *args, **kwargs)

    def debug_click(x=None, y=None, *args, **kwargs):
        """Wrapper to log all click calls"""
        # Get the call stack
        stack = traceback.extract_stack()
        caller = stack[-2]  # The function that called click

        # Log to a file
        with open("/tmp/jarvis_coordinate_debug.log", "a") as f:
            f.write(f"\n{'='*60}\n")
            if x is not None and y is not None:
                f.write(f"PyAutoGUI.click({x}, {y})\n")
            else:
                f.write(f"PyAutoGUI.click() at current position\n")
            f.write(f"Called from: {caller.filename}:{caller.lineno} in {caller.name}\n")
            f.write(f"Code: {caller.line}\n")

            # Check if coordinates are doubled
            if x and x > 2000:
                f.write(f"⚠️ WARNING: X coordinate {x} seems too large!\n")
                f.write(f"   Might be doubled from {x//2}\n")
            if y and y > 1000:
                f.write(f"⚠️ WARNING: Y coordinate {y} seems too large!\n")
                f.write(f"   Might be doubled from {y//2}\n")

        # Print to console too
        if x is not None and y is not None:
            print(f"\n🖱️ DEBUG: pyautogui.click({x}, {y})")
        else:
            print(f"\n🖱️ DEBUG: pyautogui.click() at current")
        print(f"   Called from: {caller.name} in {caller.filename}:{caller.lineno}")

        # Call original
        return original_click(x, y, *args, **kwargs)

    # Replace PyAutoGUI functions
    pyautogui.moveTo = debug_moveTo
    pyautogui.click = debug_click

    print("\n" + "="*80)
    print("COORDINATE DEBUG MODE ACTIVATED")
    print("All PyAutoGUI movements will be logged to:")
    print("  /tmp/jarvis_coordinate_debug.log")
    print("="*80 + "\n")

    # Clear the log file
    with open("/tmp/jarvis_coordinate_debug.log", "w") as f:
        f.write("JARVIS Coordinate Debug Log\n")
        f.write("="*80 + "\n")

if __name__ == "__main__":
    # Test the debug injection
    import pyautogui

    debug_inject()

    print("Testing debug mode...")
    pyautogui.moveTo(1236, 12)
    pyautogui.click(1236, 12)

    print("\nTesting with large coordinates...")
    pyautogui.moveTo(2475, 15)

    print("\nCheck /tmp/jarvis_coordinate_debug.log for details")