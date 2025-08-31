#!/usr/bin/env python3
"""Test vision processing speed"""

import asyncio
import time
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from chatbots.claude_vision_chatbot import ClaudeVisionChatbot

async def test_vision_speed():
    """Test the actual vision processing speed"""
    
    # Initialize the chatbot
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set")
        return
        
    print("Initializing Claude Vision Chatbot...")
    chatbot = ClaudeVisionChatbot(api_key=api_key)
    
    # Enable logging to see timing details
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Test the vision command
    test_command = "can you see my screen?"
    print(f"\nTesting command: '{test_command}'")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        response = await chatbot.analyze_screen_with_vision(test_command)
        
        total_time = time.time() - start_time
        print(f"\nTotal time: {total_time:.2f} seconds")
        print(f"\nResponse: {response[:200]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_vision_speed())