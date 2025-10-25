#!/usr/bin/env python3
"""
Control Center Position Setup Wizard
Finds and saves the exact position of your Control Center icon
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pyautogui
import json
import asyncio
from PIL import Image, ImageDraw
import subprocess

async def setup_control_center():
    """Interactive setup to find Control Center position"""
    
    print("\n" + "="*70)
    print("🎯 Control Center Position Setup Wizard")
    print("="*70)
    
    print("\nThis wizard will help JARVIS learn where your Control Center icon is.")
    print("We'll do this in 3 steps:\n")
    
    # Step 1: Show current position
    screen_width, screen_height = pyautogui.size()
    current_x = screen_width - 70
    current_y = 12
    
    print(f"Step 1: Current Settings")
    print(f"  Screen: {screen_width}x{screen_height}")
    print(f"  Current click position: ({current_x}, {current_y})")
    
    # Step 2: Capture and show
    print(f"\nStep 2: Capturing your menu bar...")
    
    temp_dir = Path.home() / '.jarvis' / 'screenshots' / 'setup'
    temp_dir.mkdir(parents=True, exist_ok=True)
    screenshot_path = temp_dir / 'menubar.png'
    
    # Capture just the menu bar (top 30px)
    process = await asyncio.create_subprocess_exec(
        'screencapture', '-R', f'0,0,{screen_width},30', '-x', str(screenshot_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()
    
    if screenshot_path.exists():
        # Draw marker at current position
        img = Image.open(screenshot_path)
        draw = ImageDraw.Draw(img)
        
        # Draw RED circle at current heuristic position
        marker_size = 8
        draw.ellipse(
            [current_x - marker_size, current_y - marker_size,
             current_x + marker_size, current_y + marker_size],
            outline='red', width=2
        )
        
        marked_path = temp_dir / 'menubar_marked.png'
        img.save(marked_path)
        
        print(f"  ✅ Screenshot saved: {marked_path}")
        print(f"  📸 RED circle shows where JARVIS currently clicks")
        
        # Open the image
        subprocess.run(['open', str(marked_path)])
    
    # Step 3: Get correct position
    print(f"\nStep 3: Find the Control Center Icon")
    print(f"  Look at the screenshot that just opened.")
    print(f"  The RED circle shows where JARVIS currently clicks.")
    print(f"")
    
    choice = input("Is the RED circle on the Control Center icon? (y/n): ").lower()
    
    if choice == 'y':
        print(f"\n✅ Great! Current position is correct: ({current_x}, {current_y})")
        final_x, final_y = current_x, current_y
    else:
        print(f"\nOption A: Manual Entry")
        print(f"  Move your mouse to the CENTER of the Control Center icon.")
        print(f"  Then press ENTER here...")
        input("  Press ENTER when mouse is positioned: ")
        
        # Get mouse position
        final_x, final_y = pyautogui.position()
        
        print(f"\n✅ Got it! Control Center is at: ({final_x}, {final_y})")
    
    # Step 4: Save to config
    config_path = Path(__file__).parent.parent / 'config' / 'vision_navigator_config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update the control center position
        if 'ui_elements' not in config:
            config['ui_elements'] = {}
        if 'control_center' not in config['ui_elements']:
            config['ui_elements']['control_center'] = {}
        
        # Calculate offset from right
        offset_from_right = screen_width - final_x
        
        config['ui_elements']['control_center'].update({
            'absolute_x': final_x,
            'absolute_y': final_y,
            'offset_from_right': offset_from_right,
            'screen_width': screen_width,
            'screen_height': screen_height
        })
        
        # Save updated config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"\n✅ Saved to config: {config_path}")
        print(f"   Position: ({final_x}, {final_y})")
        print(f"   Offset from right: {offset_from_right}px")
        
    except Exception as e:
        print(f"\n⚠️  Could not save to config: {e}")
        print(f"   Manual update needed:")
        print(f"   Position: ({final_x}, {final_y})")
    
    # Step 5: Test it
    print(f"\nStep 4: Test the Position")
    test = input("Would you like to test this position? (y/n): ").lower()
    
    if test == 'y':
        print(f"\n🎯 Moving mouse to Control Center position...")
        print(f"   Watch your mouse cursor move to ({final_x}, {final_y})")
        
        await asyncio.sleep(1)
        pyautogui.moveTo(final_x, final_y, duration=1.0)
        
        print(f"\n   Is the mouse cursor now over the Control Center icon?")
        verify = input("   (y/n): ").lower()
        
        if verify == 'y':
            print(f"\n   ✅ Perfect! Position is correct.")
        else:
            print(f"\n   ⚠️  Position needs adjustment.")
            print(f"   Run this script again to refine it.")
    
    print(f"\n" + "="*70)
    print("✅ Setup Complete!")
    print("="*70)
    print(f"\nNow restart your backend and try:")
    print(f"  'connect to my living room tv'")
    print(f"\nJARVIS will click at ({final_x}, {final_y}) to open Control Center")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(setup_control_center())
