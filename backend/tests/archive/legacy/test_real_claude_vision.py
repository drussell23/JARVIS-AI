#!/usr/bin/env python3
"""
Test the pure intelligence system with real Claude Vision API
"""

import asyncio
import sys
import os
import logging

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

async def test_real_claude():
    print("\n🧪 Testing Pure Intelligence with Real Claude Vision API\n")
    
    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print(f"✅ API Key found: {api_key[:20]}...")
    else:
        print("❌ No API key found!")
        return
        
    # Test the vision command handler
    from api.vision_command_handler import vision_command_handler
    
    # Initialize with the API key
    await vision_command_handler.initialize_intelligence(api_key)
    
    # Test battery query
    print("\n🔋 Testing battery query...")
    result = await vision_command_handler.handle_command("can you see my battery percentage?")
    
    print(f"\n📊 Results:")
    print(f"Response: {result.get('response')}")
    print(f"Pure Intelligence: {result.get('pure_intelligence')}")
    print(f"Error: {result.get('error', False)}")
    
    # Check if using mock or real responses
    if hasattr(vision_command_handler, 'intelligence') and vision_command_handler.intelligence:
        if vision_command_handler.intelligence.claude:
            print("\n✅ Using real Claude Vision API!")
        else:
            print("\n⚠️  Using mock responses (Claude client not initialized)")
    
    # Test another query
    print("\n🖥️  Testing screen query...")
    result2 = await vision_command_handler.handle_command("what applications do I have open?")
    print(f"Response: {result2.get('response')}")
    
if __name__ == "__main__":
    asyncio.run(test_real_claude())