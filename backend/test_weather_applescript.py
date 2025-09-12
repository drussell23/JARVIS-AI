#!/usr/bin/env python3
"""Test Weather app with various AppleScript approaches"""

import asyncio
import subprocess

async def test_weather_applescript():
    """Test different AppleScript methods to control Weather app"""
    print("🍎 Testing Weather App AppleScript Control")
    print("="*60)
    
    from system_control.macos_controller import MacOSController
    controller = MacOSController()
    
    # Ensure Weather is open
    subprocess.run(['open', '-a', 'Weather'], check=False)
    await asyncio.sleep(2)
    
    # Test 1: Check if Weather has scriptable elements
    print("\n1. Checking Weather app scriptability...")
    script = '''
    tell application "Weather"
        properties
    end tell
    '''
    success, result = controller.execute_applescript(script)
    print(f"   Weather properties: {result[:100] if success else 'Not scriptable'}")
    
    # Test 2: Try System Events with detailed UI hierarchy
    print("\n2. Exploring Weather UI hierarchy...")
    script = '''
    tell application "System Events"
        tell process "Weather"
            tell window 1
                -- Get UI elements
                set uiElements to UI elements
                set elementNames to {}
                repeat with elem in uiElements
                    set end of elementNames to (class of elem as string)
                end repeat
                return elementNames
            end tell
        end tell
    end tell
    '''
    success, result = controller.execute_applescript(script)
    if success:
        print(f"   Window UI elements: {result}")
    
    # Test 3: Try to find the location list
    print("\n3. Finding location list...")
    script = '''
    tell application "System Events"
        tell process "Weather"
            tell window 1
                -- Look for the sidebar with locations
                if exists scroll area 1 of splitter group 1 then
                    tell scroll area 1 of splitter group 1
                        if exists table 1 of scroll area 1 then
                            tell table 1 of scroll area 1
                                -- Get all rows (locations)
                                set locationCount to count of rows
                                set locationNames to {}
                                repeat with i from 1 to locationCount
                                    try
                                        set locationName to value of static text 1 of UI element 1 of row i
                                        set end of locationNames to locationName
                                    end try
                                end repeat
                                return "Found " & locationCount & " locations: " & (locationNames as string)
                            end tell
                        else
                            return "No table found"
                        end if
                    end tell
                else
                    return "No scroll area found"
                end if
            end tell
        end tell
    end tell
    '''
    success, result = controller.execute_applescript(script)
    if success:
        print(f"   Location list: {result}")
    
    # Test 4: Try to select Toronto specifically
    print("\n4. Attempting to select Toronto...")
    script = '''
    tell application "System Events"
        tell process "Weather"
            tell window 1
                tell scroll area 1 of splitter group 1
                    tell table 1 of scroll area 1
                        -- Look for Toronto
                        set rowCount to count of rows
                        repeat with i from 1 to rowCount
                            try
                                set rowText to value of static text 1 of UI element 1 of row i
                                if rowText contains "Toronto" or rowText contains "My Location" then
                                    -- Found Toronto, try to select it
                                    select row i
                                    delay 0.5
                                    
                                    -- Click it
                                    click row i
                                    delay 0.5
                                    
                                    -- Double-click for good measure
                                    click row i
                                    
                                    return "Selected Toronto at row " & i
                                end if
                            end try
                        end repeat
                        return "Toronto not found in " & rowCount & " rows"
                    end tell
                end tell
            end tell
        end tell
    end tell
    '''
    success, result = controller.execute_applescript(script)
    print(f"   Selection result: {result}")
    
    # Test 5: Try AXPress action
    print("\n5. Testing AXPress action...")
    script = '''
    tell application "System Events"
        tell process "Weather"
            tell window 1
                tell scroll area 1 of splitter group 1
                    tell table 1 of scroll area 1
                        tell row 1  -- Assuming Toronto is first
                            perform action "AXPress"
                            return "Pressed row 1"
                        end tell
                    end tell
                end tell
            end tell
        end tell
    end tell
    '''
    success, result = controller.execute_applescript(script)
    print(f"   AXPress result: {result}")
    
    print("\n" + "="*60)
    print("Analysis:")
    print("Weather app UI can be accessed via System Events")
    print("but may have special handling for location selection.")

if __name__ == "__main__":
    import os
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(test_weather_applescript())