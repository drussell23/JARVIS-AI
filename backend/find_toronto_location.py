#!/usr/bin/env python3
"""Find Toronto/My Location in Weather app sidebar"""

import asyncio
import os
import subprocess

async def find_toronto():
    """Find exact location of Toronto in Weather app"""
    print("🔍 Finding Toronto/My Location in Weather App")
    print("="*60)
    
    from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
    from system_control.macos_controller import MacOSController
    
    vision = ClaudeVisionAnalyzer(os.getenv('ANTHROPIC_API_KEY'))
    controller = MacOSController()
    
    # Open Weather
    print("\n1. Opening Weather app...")
    controller.open_application("Weather")
    await asyncio.sleep(3)
    
    # Capture screen
    print("\n2. Capturing screen...")
    screenshot = await vision.capture_screen()
    
    # Ask Claude to find Toronto/My Location
    print("\n3. Looking for Toronto/My Location...")
    result = await vision.analyze_screenshot_async(
        screenshot,
        """Look at the Weather app sidebar on the left. Find 'My Location' or 'Toronto' in the list.
        Tell me:
        1. What is the exact text shown for Toronto/My Location?
        2. What position is it in the list? (1st, 2nd, 3rd, etc)
        3. What other cities are visible in the sidebar?
        4. Is 'My Location' at the very top or somewhere else?""",
        quick_mode=True
    )
    
    print(f"\nAnalysis: {result.get('description', 'Unknown')}")
    
    # Now try to click on the correct position
    print("\n4. Determining click coordinates...")
    description = result.get('description', '').lower()
    
    # Adjust Y coordinate based on position
    base_y = 150  # First item
    item_height = 40  # Approximate height between items
    
    if 'first' in description or 'top' in description:
        click_y = base_y
    elif 'second' in description:
        click_y = base_y + item_height
    elif 'third' in description:
        click_y = base_y + (item_height * 2)
    else:
        click_y = base_y  # Default to first
        
    print(f"\n5. Clicking at position (125, {click_y})...")
    await controller.click_at(125, click_y)
    await asyncio.sleep(2)
    
    # Verify
    print("\n6. Verifying selection...")
    weather_result = await vision.analyze_weather_fast()
    
    if weather_result.get('success'):
        analysis = weather_result.get('analysis', '')
        print(f"\n✅ Now showing: {analysis}")
        
        if 'toronto' in analysis.lower():
            print("\n🎉 SUCCESS: Found and selected Toronto!")
        else:
            print("\n⚠️  Still need to adjust coordinates")

if __name__ == "__main__":
    os.chdir('/Users/derekjrussell/Documents/repos/JARVIS-AI-Agent/backend')
    asyncio.run(find_toronto())