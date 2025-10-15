#!/usr/bin/env python3
"""Test chatbot monitoring directly"""

import asyncio
import os

async def test_chatbot():
    """Test the chatbot directly"""
    from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
    
    # Create chatbot
    api_key = os.getenv('ANTHROPIC_API_KEY')
    chatbot = ClaudeVisionChatbot(api_key)
    
    print(f"✅ Chatbot created")
    print(f"📷 Vision analyzer: {chatbot.vision_analyzer is not None}")
    
    if chatbot.vision_analyzer:
        print(f"   Type: {type(chatbot.vision_analyzer).__name__}")
        print(f"   Video streaming enabled: {chatbot.vision_analyzer.config.enable_video_streaming}")
        
        # Test monitoring command
        print("\n🔍 Testing monitoring command...")
        response = await chatbot.generate_response("start monitoring my screen")
        print(f"\n📨 Response: {response}")
        
        if "Failed to start" in response:
            # Check video streaming status
            vs = await chatbot.vision_analyzer.get_video_streaming()
            print(f"\n🎥 Video streaming module: {vs is not None}")
            if vs:
                print(f"   Is capturing: {vs.is_capturing}")

if __name__ == "__main__":
    asyncio.run(test_chatbot())