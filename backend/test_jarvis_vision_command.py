#!/usr/bin/env python3
"""
Test JARVIS vision command through the full pipeline
"""

import asyncio
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from system_control.vision_action_handler import get_vision_action_handler

async def test_jarvis_vision():
    print("🧪 Testing JARVIS Vision Command Pipeline")
    print("=" * 50)
    
    # Get the vision handler
    handler = get_vision_action_handler()
    
    print("\n1️⃣ Testing describe_screen through JARVIS...")
    
    # Test with explicit query
    params = {
        'query': 'describe what you see on my screen in detail'
    }
    
    result = await handler.describe_screen(params)
    
    print(f"\n✅ Success: {result.success}")
    print(f"📊 Confidence: {result.confidence}")
    print(f"\n🤖 JARVIS Response:")
    print("-" * 50)
    print(result.description)
    print("-" * 50)
    
    if result.data:
        print(f"\n📈 Metadata:")
        for key, value in result.data.items():
            print(f"  • {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_jarvis_vision())