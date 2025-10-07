#!/usr/bin/env python3
"""
Test script to verify vision command fix
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


async def test_vision_fix():
    """Test the vision command fix"""
    print("🔍 Testing Vision Command Fix")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return
    print(f"✅ API Key loaded: {api_key[:20]}...")

    try:
        # Test 1: Create JARVIS agent with vision analyzer
        print("\n1️⃣ Creating JARVIS Agent with Vision...")
        from vision.claude_vision_analyzer_main import ClaudeVisionAnalyzer
        from chatbots.claude_vision_chatbot import ClaudeVisionChatbot
        from voice.jarvis_agent_voice import JARVISAgentVoice

        # Create vision analyzer
        vision_analyzer = ClaudeVisionAnalyzer(api_key)
        print("✅ Vision analyzer created")

        # Create JARVIS agent with vision analyzer
        jarvis = JARVISAgentVoice("Sir", vision_analyzer=vision_analyzer)
        print("✅ JARVIS agent created with vision analyzer")

        # Test 2: Test vision command through async pipeline
        print("\n2️⃣ Testing Vision Command Through Pipeline...")
        test_command = "can you see my screen"
        print(f"   Command: '{test_command}'")

        response = await jarvis.process_voice_input(test_command)
        print(f"   Response: {response}")

        # Test 3: Test different vision commands
        print("\n3️⃣ Testing Different Vision Commands...")
        vision_commands = [
            "what do you see on my screen",
            "what's on my screen",
            "describe my screen",
            "analyze my screen",
        ]

        for cmd in vision_commands:
            print(f"\n   Testing: '{cmd}'")
            try:
                response = await jarvis.process_voice_input(cmd)
                print(f"   Response: {response[:200]}...")
            except Exception as e:
                print(f"   Error: {e}")

        print("\n✅ Vision command fix test completed!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_vision_fix())

