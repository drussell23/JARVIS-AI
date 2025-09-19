#!/usr/bin/env python3
"""Test that multi-space vision is working after fixes"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_terminal_query():
    """Test the 'Where is Terminal?' query"""
    from api.pure_vision_intelligence import PureVisionIntelligence
    from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
    import numpy as np
    import os
    
    # Check if we have an API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠️  No ANTHROPIC_API_KEY found. Using mock mode.")
        # Create mock client
        claude_client = type('MockClient', (), {
            'analyze_image_with_prompt': lambda self, **kwargs: asyncio.coroutine(lambda: {
                'text': "I can see your desktop.",
                'detailed_description': "Desktop visible"
            })(),
            'analyze_multiple_images_with_prompt': lambda self, **kwargs: asyncio.coroutine(lambda: {
                'text': "Looking across your desktop spaces, I can see Terminal is running on Desktop 2.",
                'detailed_description': "Terminal found on Desktop 2"
            })()
        })()
    else:
        print("✅ API key found. Using real Claude Vision.")
        claude_client = ClaudeVisionChatbot(api_key=api_key)
    
    try:
        # Create vision intelligence
        vision = PureVisionIntelligence(claude_client, enable_multi_space=True)
        
        # Create mock screenshot
        screenshot = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Test queries
        test_queries = [
            "Where is Terminal?",
            "Show me all my workspaces",
            "What's on Desktop 2?"
        ]
        
        for query in test_queries:
            print(f"\n💬 User: {query}")
            try:
                response = await vision.understand_and_respond(screenshot, query)
                print(f"🤖 JARVIS: {response[:200]}...")
                
                # Check multi-space context if available
                if hasattr(vision, '_last_multi_space_context') and vision._last_multi_space_context:
                    ctx = vision._last_multi_space_context
                    print(f"   📊 Multi-space context: {ctx.analyzed_spaces}/{ctx.total_spaces} spaces analyzed")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 Testing Multi-Space Vision System Fix")
    print("=" * 50)
    asyncio.run(test_terminal_query())