#!/usr/bin/env python3
"""
Direct test of Claude Vision to ensure it's working
"""

import os
import asyncio
from vision.screen_capture_fallback import capture_with_intelligence

async def test_claude_vision():
    print("🧪 Testing Claude Vision Analysis")
    print("=" * 50)
    
    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found in environment")
        print("Please set: export ANTHROPIC_API_KEY=your-key-here")
        return
    
    print("✅ Claude API key found")
    
    # Test with specific query about what's on screen
    query = "Please describe in detail what you see on the screen. Include any applications, windows, file icons, or other visual elements. Be specific about what you observe."
    
    print(f"\n📸 Capturing screen and sending to Claude...")
    print(f"Query: {query}")
    
    result = capture_with_intelligence(query=query, use_claude=True)
    
    print("\n📊 Results:")
    print(f"Success: {result.get('success')}")
    print(f"Intelligence Used: {result.get('intelligence_used')}")
    
    if result.get('analysis'):
        print(f"\n🤖 Claude's Analysis:")
        print("-" * 50)
        print(result['analysis'])
        print("-" * 50)
    else:
        print(f"\n❌ Error: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    asyncio.run(test_claude_vision())