#!/usr/bin/env python3
"""Basic test of JARVIS vision functionality"""

import asyncio
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic():
    """Test basic vision functionality"""
    from claude_vision_analyzer import ClaudeVisionAnalyzer
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("Please set ANTHROPIC_API_KEY")
        return
    
    # Initialize
    jarvis = ClaudeVisionAnalyzer(api_key)
    
    print("\n🧪 Testing basic vision...")
    
    # Test 1: Can we capture screen?
    try:
        screenshot = await jarvis.capture_screen()
        if screenshot:
            print("✅ Screen capture working")
        else:
            print("❌ Screen capture failed")
    except Exception as e:
        print(f"❌ Screen capture error: {e}")
    
    # Test 2: Can we analyze?
    try:
        context = await jarvis.get_screen_context()
        if context and 'description' in context:
            print(f"✅ Vision analysis working")
            print(f"   I can see: {context['description'][:100]}...")
        else:
            print(f"❌ Vision analysis failed: {context}")
    except Exception as e:
        print(f"❌ Vision analysis error: {e}")
    
    # Test 3: Can we respond to commands?
    try:
        response = await jarvis.see_and_respond("What do you see?")
        if response.get('success'):
            print(f"✅ Command response working")
            print(f"   Response: {response['response'][:100]}...")
        else:
            print(f"❌ Command response failed: {response}")
    except Exception as e:
        print(f"❌ Command response error: {e}")
    
    # Cleanup
    await jarvis.cleanup_all_components()
    print("\n✅ Basic test complete")

if __name__ == "__main__":
    asyncio.run(test_basic())