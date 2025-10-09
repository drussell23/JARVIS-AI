#!/usr/bin/env python3
"""
Quick test for battery percentage query
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_battery():
    from api.vision_command_handler import vision_command_handler
    
    # Initialize
    await vision_command_handler.initialize_intelligence()
    
    # Test battery query
    print("Testing: 'can you see my battery percentage?'")
    result = await vision_command_handler.handle_command("can you see my battery percentage?")
    
    print(f"\nResponse: {result.get('response')}")
    print(f"Pure Intelligence: {result.get('pure_intelligence')}")
    print(f"Success: {not result.get('error')}")
    
if __name__ == "__main__":
    asyncio.run(test_battery())