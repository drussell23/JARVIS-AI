#!/usr/bin/env python3
"""
Test that monitoring commands give concise responses
"""

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.vision_command_handler import vision_command_handler

async def test_concise_responses():
    """Test that monitoring commands give brief, clear responses"""
    print("\n🧪 TESTING CONCISE MONITORING RESPONSES")
    print("=" * 60)
    
    # Initialize
    print("Initializing vision command handler...")
    await vision_command_handler.initialize_intelligence()
    
    # Test start monitoring
    print("\n1️⃣ Testing 'start monitoring my screen':")
    result = await vision_command_handler.handle_command("start monitoring my screen")
    response = result.get('response', 'No response')
    
    print(f"\nResponse length: {len(response)} characters")
    print(f"Response: {response}")
    
    # Count sentences
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    print(f"Number of sentences: {len(sentences)}")
    
    if len(response) < 200:
        print("✅ Response is concise!")
    else:
        print("⚠️ Response is too long")
    
    # Wait a bit
    await asyncio.sleep(3)
    
    # Test stop monitoring
    print("\n2️⃣ Testing 'stop monitoring':")
    result = await vision_command_handler.handle_command("stop monitoring")
    response = result.get('response', 'No response')
    
    print(f"\nResponse length: {len(response)} characters")
    print(f"Response: {response}")
    
    if len(response) < 100:
        print("✅ Response is concise!")
    else:
        print("⚠️ Response is too long")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    asyncio.run(test_concise_responses())