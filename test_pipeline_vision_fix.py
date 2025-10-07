#!/usr/bin/env python3
"""
Test script to verify async pipeline vision command fix
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))


async def test_pipeline_vision_fix():
    """Test the async pipeline vision command fix"""
    print("🔍 Testing Async Pipeline Vision Command Fix")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return
    print(f"✅ API Key loaded: {api_key[:20]}...")

    try:
        # Test 1: Create vision analyzer and chatbot
        print("\n1️⃣ Creating Vision Components...")
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from chatbots.claude_vision_chatbot import ClaudeVisionChatbot

        # Create vision analyzer
        vision_analyzer = ClaudeVisionAnalyzer(api_key)
        print("✅ Vision analyzer created")

        # Create chatbot with vision analyzer
        chatbot = ClaudeVisionChatbot(api_key, vision_analyzer=vision_analyzer)
        print("✅ Claude chatbot created with vision analyzer")

        # Test 2: Test async pipeline directly
        print("\n2️⃣ Testing Async Pipeline Directly...")
        from core.async_pipeline import get_async_pipeline

        # Create a mock JARVIS instance with vision capabilities
        class MockJARVIS:
            def __init__(self, chatbot):
                self.claude_chatbot = chatbot

            async def _handle_vision_command(self, text):
                """Mock vision command handler"""
                print(f"   [MOCK] Handling vision command: {text}")
                return await self.claude_chatbot.analyze_screen_with_vision(text)

        mock_jarvis = MockJARVIS(chatbot)

        # Get async pipeline
        pipeline = get_async_pipeline(mock_jarvis)
        print("✅ Async pipeline created")

        # Test 3: Test vision commands through pipeline
        print("\n3️⃣ Testing Vision Commands Through Pipeline...")
        vision_commands = [
            "can you see my screen",
            "what do you see on my screen",
            "what's on my screen",
        ]

        for cmd in vision_commands:
            print(f"\n   Testing: '{cmd}'")
            try:
                result = await pipeline.process_async(cmd, "Sir")
                print(
                    f"   Intent: {result.get('metadata', {}).get('intent', 'unknown')}"
                )
                print(f"   Response: {result.get('response', 'No response')[:200]}...")
            except Exception as e:
                print(f"   Error: {e}")

        print("\n✅ Pipeline vision command fix test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_pipeline_vision_fix())

