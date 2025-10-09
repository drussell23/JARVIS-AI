#!/usr/bin/env python3
"""
Test script to debug vision analysis hanging issue
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_vision_analysis():
    """Test the vision analysis flow directly"""
    
    print("Testing vision analysis...")
    
    # Import the vision command handler
    from api.vision_command_handler import vision_command_handler
    
    # Test 1: Check if monitoring is active
    print("\n1. Checking if screen monitoring is active...")
    monitoring_result = await vision_command_handler.handle_command("start monitoring my screen")
    print(f"Monitoring result: {monitoring_result}")
    
    # Wait a bit for monitoring to start
    await asyncio.sleep(2)
    
    # Test 2: Try to analyze the screen
    print("\n2. Testing screen analysis...")
    analysis_result = await vision_command_handler.handle_command("can you see my screen?")
    print(f"Analysis result handled: {analysis_result.get('handled')}")
    print(f"Response length: {len(analysis_result.get('response', ''))}")
    print(f"Response preview: {analysis_result.get('response', '')[:200]}...")
    
    # Test 3: Stop monitoring
    print("\n3. Stopping monitoring...")
    stop_result = await vision_command_handler.handle_command("stop monitoring my screen")
    print(f"Stop result: {stop_result}")

if __name__ == "__main__":
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set!")
        print("Please set it with: export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)
    
    # Run the test
    asyncio.run(test_vision_analysis())