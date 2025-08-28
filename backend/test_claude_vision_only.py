#!/usr/bin/env python3
"""
Test script to verify Claude Vision integration is working properly
This will help debug why you're still getting generic responses
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

# Test imports
print("Testing Claude Vision Integration...")
print("=" * 60)

# Check API key
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    print("❌ ANTHROPIC_API_KEY not found in environment!")
    print("Please set: export ANTHROPIC_API_KEY='your-key-here'")
    sys.exit(1)
else:
    print("✅ Anthropic API key found")

async def test_vision_components():
    """Test each component of the vision system"""
    
    print("\n1. Testing screen_capture_fallback.py...")
    try:
        from vision.screen_capture_fallback import capture_with_intelligence
        
        # Test basic capture
        result = capture_with_intelligence(use_claude=False)
        if result["success"]:
            print("✅ Screen capture working")
        else:
            print("❌ Screen capture failed:", result.get("error"))
            
        # Test Claude analysis
        print("\n2. Testing Claude Vision analysis...")
        result = capture_with_intelligence(
            query="What applications are open and what is the user working on?",
            use_claude=True
        )
        
        if result.get("intelligence_used") and result.get("analysis"):
            print("✅ Claude Vision working!")
            print(f"Analysis: {result['analysis'][:200]}...")
        else:
            print("❌ Claude Vision failed")
            
    except Exception as e:
        print(f"❌ Error in screen_capture_fallback: {e}")
        
    print("\n3. Testing intelligent_vision_integration.py...")
    try:
        from vision.intelligent_vision_integration import IntelligentJARVISVision
        
        vision = IntelligentJARVISVision()
        
        # Test the main command
        response = await vision.handle_intelligent_command("can you see my screen?")
        print(f"Response: {response}")
        
        if "I can read" in response and "text elements" in response:
            print("❌ Still getting generic response!")
        elif "I can see your screen" in response and api_key:
            print("✅ Getting proper Claude response")
        
    except Exception as e:
        print(f"❌ Error in intelligent_vision_integration: {e}")
        
    print("\n4. Testing screen_vision.py...")
    try:
        from vision.screen_vision import ScreenVisionSystem
        
        vision_system = ScreenVisionSystem()
        
        # Check if Claude analyzer is initialized
        if vision_system.claude_analyzer:
            print("✅ Claude analyzer initialized in ScreenVisionSystem")
        else:
            print("❌ Claude analyzer NOT initialized in ScreenVisionSystem")
            
        # Test capture_and_describe
        response = await vision_system.capture_and_describe()
        print(f"capture_and_describe response: {response[:200]}...")
        
        if "I can read" in response and "text elements" in response:
            print("❌ capture_and_describe still has generic response!")
        
    except Exception as e:
        print(f"❌ Error in screen_vision: {e}")

if __name__ == "__main__":
    print("\nRunning vision system tests...\n")
    asyncio.run(test_vision_components())
    print("\n" + "=" * 60)
    print("Test complete. Check the results above.")