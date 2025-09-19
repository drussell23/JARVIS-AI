#!/usr/bin/env python3
"""
Debug vision response flow
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

async def test_vision_flow():
    # Test the vision command handler directly
    from api.vision_command_handler import vision_command_handler
    
    # Initialize with API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    await vision_command_handler.initialize_intelligence(api_key)
    
    # Test battery query
    print("\n🔋 Testing vision command handler directly...")
    result = await vision_command_handler.handle_command("can you see my battery percentage?")
    
    print(f"\n📊 Raw result type: {type(result)}")
    print(f"📊 Raw result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
    print(f"\n📊 Response field: {result.get('response')}")
    print(f"📊 Response type: {type(result.get('response'))}")
    
    # Now test through jarvis_voice_api
    print("\n\n🎤 Testing through jarvis_voice_api...")
    from api.jarvis_voice_api import jarvis_api
    from api.jarvis_voice_api import JarvisCommand
    
    cmd = JarvisCommand(text="can you see my battery percentage?", confidence=1.0)
    voice_result = await jarvis_api.process_command(cmd)
    
    print(f"\n📊 Voice result type: {type(voice_result)}")
    print(f"📊 Voice result keys: {voice_result.keys() if isinstance(voice_result, dict) else 'Not a dict'}")
    print(f"\n📊 Voice response field: {voice_result.get('response')}")
    print(f"📊 Voice response type: {type(voice_result.get('response'))}")
    
    # Check if response is a string or dict
    response = voice_result.get('response')
    if isinstance(response, dict):
        print(f"\n⚠️  Response is a dict with keys: {response.keys()}")
        print(f"📊 Content field: {response.get('content', 'No content field')}")
        print(f"📊 Raw result field: {response.get('raw_result', 'No raw_result field')}")

if __name__ == "__main__":
    asyncio.run(test_vision_flow())