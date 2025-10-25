#!/usr/bin/env python3
"""Find Control Center coordinates on macOS"""

import pyautogui

def find_control_center():
    """Find Control Center icon coordinates"""
    screen_width, screen_height = pyautogui.size()
    
    # Control Center is ~45 pixels from right edge, centered in 25px menu bar
    x = screen_width - 45
    y = 12
    
    print(f"Screen size: {screen_width}x{screen_height}")
    print(f"Control Center at: ({x}, {y})")
    print(f"\nMoving mouse there now (not clicking)...")
    
    pyautogui.moveTo(x, y, duration=0.5)
    print(f"✅ Mouse at Control Center!")
    
    return (x, y)

if __name__ == "__main__":
    coords = find_control_center()
    print(f"\nTo click it, add: pyautogui.click()")
