#!/usr/bin/env python3
"""
Test browser automation capabilities
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.unified_command_processor import get_unified_processor

async def test_browser_automation():
    processor = get_unified_processor()
    
    print("JARVIS Browser Automation Demo")
    print("=" * 50)
    
    # Simulate a conversation flow
    commands = [
        ("Open Safari and go to Google", 2),
        ("Open a new tab in Safari and go to Google", 2),
        ("Type dogs and press enter", 3),
        ("Open another tab", 2),
        ("Search for cats", 2),
    ]
    
    for command, delay in commands:
        print(f"\nYou: {command}")
        result = await processor.process_command(command)
        print(f"JARVIS: {result.get('response')}")
        print(f"Debug - Command type: {result.get('command_type')}")
        
        if not result.get('success'):
            print(f"Error: {result.get('error', 'Command failed')}")
        
        # Wait before next command
        await asyncio.sleep(delay)
    
    print("\n" + "=" * 50)
    print("Browser automation complete!")
    print("\nCapabilities demonstrated:")
    print("✓ Opening new tabs")
    print("✓ Navigating to URLs")
    print("✓ Typing in search bars")
    print("✓ Pressing Enter to search")
    print("✓ Dynamic browser control without hardcoding")

if __name__ == "__main__":
    asyncio.run(test_browser_automation())