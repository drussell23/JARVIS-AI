#!/usr/bin/env python3
"""
Test AirPlay Menu Detection
============================

Quick test to see what displays are available in Screen Mirroring menu.
"""

import asyncio
import subprocess

async def test_airplay_menu():
    print("🔍 Testing AirPlay Menu Detection...\n")
    
    # Method 1: Try to query Display menu in Control Center (macOS Ventura+)
    print("METHOD 1: Control Center Display Menu")
    script1 = """
    tell application "System Events"
        try
            tell process "ControlCenter"
                set displayMenu to menu bar item "Display" of menu bar 1
                click displayMenu
                delay 0.5
                
                set menuItems to name of every menu item of menu 1 of displayMenu
                click displayMenu
                
                return menuItems as text
            end tell
        on error errMsg
            return "Error: " & errMsg
        end try
    end tell
    """
    
    try:
        result = await asyncio.create_subprocess_exec(
            "osascript", "-e", script1,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10.0)
        output = stdout.decode().strip()
        print(f"Output: {output}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Method 2: Check for AirPlay menu extras
    print("METHOD 2: Menu Bar Extras")
    script2 = """
    tell application "System Events"
        try
            tell process "SystemUIServer"
                set menuBarItems to name of every menu bar item of menu bar 1
                return menuBarItems as text
            end tell
        on error errMsg
            return "Error: " & errMsg
        end try
    end tell
    """
    
    try:
        result = await asyncio.create_subprocess_exec(
            "osascript", "-e", script2,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10.0)
        output = stdout.decode().strip()
        print(f"Menu bar items: {output}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Method 3: Try system_profiler
    print("METHOD 3: system_profiler")
    try:
        result = await asyncio.create_subprocess_exec(
            "system_profiler", "SPDisplaysDataType", "-json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(result.communicate(), timeout=10.0)
        import json
        data = json.loads(stdout.decode())
        print(f"Displays found: {len(data.get('SPDisplaysDataType', []))}")
        for display in data.get('SPDisplaysDataType', []):
            print(f"  - {display.get('_name', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}\n")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("1. Turn on your Living Room TV")
    print("2. Make sure it's on the same Wi-Fi network as your MacBook")
    print("3. On your MacBook, click the Screen Mirroring icon in menu bar")
    print("4. Check if 'Living Room TV' appears in the menu")
    print("5. If it does, run this script again")

if __name__ == "__main__":
    asyncio.run(test_airplay_menu())
