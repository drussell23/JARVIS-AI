#!/usr/bin/env python3
"""
Quick Weather Test - Simple test to see what's happening
"""

import asyncio
import os
import sys
import time
import logging

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

async def quick_test():
    """Quick test of weather functionality"""
    print("🚀 Quick Weather Test")
    print("=" * 50)
    
    # Check API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ No API key found!")
        return
    
    # Test 1: Basic weather app opening
    print("\n1. Opening Weather app...")
    import subprocess
    result = subprocess.run(['open', '-a', 'Weather'], capture_output=True)
    if result.returncode == 0:
        print("✅ Weather app opened successfully")
    else:
        print(f"❌ Failed to open Weather app: {result.stderr}")
    
    await asyncio.sleep(2)
    
    # Test 2: Basic vision capture
    print("\n2. Testing vision capture...")
    try:
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzerMain
        analyzer = ClaudeVisionAnalyzerMain(os.getenv("ANTHROPIC_API_KEY"))
        
        start = time.time()
        screenshot = await analyzer.capture_screen()
        elapsed = time.time() - start
        
        if screenshot:
            print(f"✅ Screen captured in {elapsed:.2f}s")
        else:
            print("❌ Screen capture failed")
            
    except Exception as e:
        print(f"❌ Vision error: {e}")
        return
    
    # Test 3: Quick weather read
    print("\n3. Testing quick weather read...")
    try:
        start = time.time()
        result = await asyncio.wait_for(
            analyzer.describe_screen({
                'query': 'Look at the Weather app. What is the current temperature number?'
            }),
            timeout=10.0
        )
        elapsed = time.time() - start
        
        if result['success']:
            print(f"✅ Weather read in {elapsed:.2f}s")
            print(f"Result: {result['description'][:100]}...")
        else:
            print("❌ Weather read failed")
            
    except asyncio.TimeoutError:
        print("❌ Weather read timed out!")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(quick_test())