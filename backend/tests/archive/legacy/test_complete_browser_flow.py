#!/usr/bin/env python3
"""
Complete browser automation flow test
Demonstrates advanced browser control without hardcoding
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.unified_command_processor import get_unified_processor

async def test_complete_flow():
    processor = get_unified_processor()
    
    print("JARVIS Complete Browser Automation Demo")
    print("=" * 50)
    
    # Complete conversation flow
    commands = [
        ("Open Safari", 2),
        ("Go to Google", 2),
        ("Type artificial intelligence and press enter", 3),
        ("Open a new tab", 2),
        ("Search for weather today", 3),
        ("Open another tab and go to github", 2),
        ("Type python projects and press enter", 3),
        ("Open Chrome and go to youtube", 2),
        ("Search for music videos", 3),
    ]
    
    for i, (command, delay) in enumerate(commands):
        print(f"\n[Step {i+1}] You: {command}")
        result = await processor.process_command(command)
        print(f"JARVIS: {result.get('response')}")
        
        if not result.get('success'):
            print(f"⚠️  Error: {result.get('error', 'Command failed')}")
        else:
            print("✅ Success")
        
        # Wait before next command
        await asyncio.sleep(delay)
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nCapabilities demonstrated:")
    print("✓ Opening specific browsers (Safari, Chrome)")
    print("✓ Navigating to websites")
    print("✓ Typing in search bars")
    print("✓ Pressing Enter to search")
    print("✓ Opening new tabs")
    print("✓ Context awareness (continuing in same browser)")
    print("✓ Cross-browser support")
    print("✓ Natural language understanding")

if __name__ == "__main__":
    asyncio.run(test_complete_flow())