#!/usr/bin/env python3
"""
Debug Coordinate Detection
=========================

Debug script to see exactly where the Enhanced Vision Pipeline detects and clicks.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import pyautogui
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO)


async def debug_coordinate_detection():
    """Debug coordinate detection step by step"""
    
    print("\n" + "="*70)
    print("🔍 Debug Coordinate Detection")
    print("="*70)
    
    # Get screen info
    screen_width, screen_height = pyautogui.size()
    print(f"\n1️⃣  Screen Info:")
    print(f"   Resolution: {screen_width}x{screen_height}")
    
    # Get current mouse position
    current_pos = pyautogui.position()
    print(f"   Current mouse: {current_pos}")
    
    # Capture menu bar region
    print(f"\n2️⃣  Capturing menu bar region...")
    
    import subprocess
    temp_dir = Path.home() / '.jarvis' / 'screenshots' / 'debug'
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    menu_bar_path = temp_dir / 'menubar_debug.png'
    
    # Capture just the menu bar (top 30px)
    process = await asyncio.create_subprocess_exec(
        'screencapture', '-R', f'0,0,{screen_width},30', '-x', str(menu_bar_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()
    
    if menu_bar_path.exists():
        print(f"   ✅ Menu bar captured: {menu_bar_path}")
        
        # Load and analyze
        img = Image.open(menu_bar_path)
        print(f"   Image size: {img.size}")
        
        # Draw debug markers
        debug_img = img.copy()
        draw = ImageDraw.Draw(debug_img)
        
        # Mark Control Center area (typical position)
        cc_x = screen_width - 100  # 100px from right
        cc_y = 15  # Center of menu bar
        
        # Draw circle at expected Control Center position
        marker_size = 10
        draw.ellipse(
            [cc_x - marker_size, cc_y - marker_size,
             cc_x + marker_size, cc_y + marker_size],
            outline='red', width=3
        )
        draw.text((cc_x - 50, cc_y + 20), "Expected CC", fill='red')
        
        # Draw screen dimensions
        draw.text((10, 5), f"Screen: {screen_width}x{screen_height}", fill='yellow')
        
        # Save debug image
        debug_path = temp_dir / 'menubar_debug_marked.png'
        debug_img.save(debug_path)
        
        print(f"   📸 Debug image saved: {debug_path}")
        print(f"   🔴 Red circle shows expected Control Center position")
        
        # Open the image
        import subprocess
        subprocess.run(['open', str(debug_path)])
        
        print(f"\n3️⃣  Expected Control Center Position:")
        print(f"   X: {cc_x} (100px from right edge)")
        print(f"   Y: {cc_y} (center of menu bar)")
        print(f"   Global coordinates: ({cc_x}, {cc_y})")
        
        # Test clicking at expected position
        print(f"\n4️⃣  Testing click at expected position...")
        print(f"   Moving mouse to ({cc_x}, {cc_y})")
        
        pyautogui.moveTo(cc_x, cc_y, duration=1.0)
        
        print(f"   Is the mouse cursor now over the Control Center icon?")
        verify = input("   (y/n): ").lower()
        
        if verify == 'y':
            print(f"   ✅ Position is correct!")
            print(f"   📝 Control Center is at: ({cc_x}, {cc_y})")
        else:
            print(f"   ⚠️  Position needs adjustment")
            print(f"   Move your mouse to the Control Center icon and press ENTER...")
            input("   Press ENTER when positioned: ")
            
            # Get actual position
            actual_pos = pyautogui.position()
            print(f"   ✅ Actual Control Center position: {actual_pos}")
            
            # Calculate offset
            offset_x = actual_pos[0] - cc_x
            offset_y = actual_pos[1] - cc_y
            
            print(f"   📊 Offset from expected: ({offset_x}, {offset_y})")
            
            if abs(offset_x) > 20 or abs(offset_y) > 10:
                print(f"   ⚠️  Large offset detected - pipeline needs adjustment")
                print(f"   💡 Update region config or coordinate calculation")
            else:
                print(f"   ✅ Offset is reasonable")
    
    print(f"\n" + "="*70)
    print("✅ Debug complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(debug_coordinate_detection())