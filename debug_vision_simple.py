#!/usr/bin/env python3
"""
Simple debug script to test vision analyzer directly
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


async def test_vision_analyzer():
    """Test the vision analyzer directly"""
    print("🔍 Testing Vision Analyzer Directly")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return
    print(f"✅ API Key loaded: {api_key[:20]}...")

    try:
        # Test 1: Create vision analyzer
        print("\n1️⃣ Creating Vision Analyzer...")
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer

        analyzer = ClaudeVisionAnalyzer(api_key)
        print("✅ Vision analyzer created")

        # Test 2: Test Claude chatbot
        print("\n2️⃣ Creating Claude Chatbot...")
        from chatbots.claude_vision_chatbot import ClaudeVisionChatbot

        chatbot = ClaudeVisionChatbot(api_key, vision_analyzer=analyzer)
        print("✅ Claude chatbot created")
        print(f"   - Vision analyzer attached: {chatbot.vision_analyzer is not None}")

        # Test 3: Test screen analysis
        print("\n3️⃣ Testing Screen Analysis...")
        test_command = "can you see my screen"
        print(f"   Command: '{test_command}'")

        response = await chatbot.analyze_screen_with_vision(test_command)
        print(f"   Response: {response}")

        # Test 4: Test simple vision command
        print("\n4️⃣ Testing Simple Vision Command...")
        simple_response = await chatbot.generate_response(
            "What do you see on my screen?"
        )
        print(f"   Simple response: {simple_response}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_vision_analyzer())

