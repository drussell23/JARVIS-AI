#!/usr/bin/env python3
"""Debug JARVIS monitoring issue"""

import asyncio
import os
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_jarvis_monitoring():
    """Test JARVIS monitoring directly"""
    
    # First test chatbot directly
    logger.info("=== Testing Chatbot Directly ===")
    from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
    api_key = os.getenv('ANTHROPIC_API_KEY')
    chatbot = ClaudeVisionChatbot(api_key)
    
    response = await chatbot.generate_response("start monitoring my screen")
    logger.info(f"Direct chatbot response: {response[:200]}...")
    
    # Now test through JARVIS
    logger.info("\n=== Testing Through JARVIS ===")
    from api.jarvis_factory import create_jarvis_agent, set_app_state, get_vision_analyzer
    
    # Create a mock app state with vision analyzer
    class MockAppState:
        def __init__(self):
            from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
            self.vision_analyzer = ClaudeVisionAnalyzer(api_key)
    
    app_state = MockAppState()
    set_app_state(app_state)
    
    # Check if vision analyzer is available
    vision_analyzer = get_vision_analyzer()
    logger.info(f"Vision analyzer from factory: {vision_analyzer is not None}")
    
    # Create JARVIS agent
    jarvis = create_jarvis_agent()
    logger.info(f"JARVIS created: {jarvis is not None}")
    logger.info(f"JARVIS has chatbot: {hasattr(jarvis, 'claude_chatbot')}")
    if hasattr(jarvis, 'claude_chatbot'):
        logger.info(f"JARVIS chatbot has vision analyzer: {jarvis.claude_chatbot.vision_analyzer is not None}")
    
    # Process command through JARVIS
    jarvis.running = True
    response = await jarvis.process_voice_input("start monitoring my screen")
    logger.info(f"JARVIS response: {response[:200]}...")

if __name__ == "__main__":
    asyncio.run(test_jarvis_monitoring())