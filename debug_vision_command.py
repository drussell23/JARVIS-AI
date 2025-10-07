#!/usr/bin/env python3
"""
Debug script to test JARVIS vision command handling
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


async def test_vision_command():
    """Test the vision command handling"""
    print("🔍 Testing JARVIS Vision Command Handling")
    print("=" * 60)

    # Check API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not found")
        return
    print(f"✅ API Key loaded: {api_key[:20]}...")

    try:
        # Test 1: Create JARVIS agent
        print("\n1️⃣ Creating JARVIS Agent...")
        from api.jarvis_factory import create_jarvis_agent

        jarvis = create_jarvis_agent("Sir")
        print(f"✅ JARVIS agent created")
        print(
            f"   - Claude chatbot: {hasattr(jarvis, 'claude_chatbot') and jarvis.claude_chatbot is not None}"
        )
        print(
            f"   - Vision analyzer: {hasattr(jarvis, 'vision_analyzer') and jarvis.vision_analyzer is not None}"
        )

        if hasattr(jarvis, "claude_chatbot") and jarvis.claude_chatbot:
            print(
                f"   - Chatbot vision analyzer: {hasattr(jarvis.claude_chatbot, 'vision_analyzer') and jarvis.claude_chatbot.vision_analyzer is not None}"
            )

        # Test 2: Test vision command
        print("\n2️⃣ Testing Vision Command...")
        test_command = "can you see my screen"
        print(f"   Command: '{test_command}'")

        response = await jarvis.process_voice_input(test_command)
        print(f"   Response: {response}")

        # Test 3: Check if vision analyzer is working
        print("\n3️⃣ Testing Vision Analyzer Directly...")
        if (
            hasattr(jarvis, "claude_chatbot")
            and jarvis.claude_chatbot
            and hasattr(jarvis.claude_chatbot, "vision_analyzer")
            and jarvis.claude_chatbot.vision_analyzer
        ):
            try:
                # Test direct vision analysis
                vision_response = (
                    await jarvis.claude_chatbot.analyze_screen_with_vision(
                        "What do you see on my screen?"
                    )
                )
                print(f"   Direct vision response: {vision_response[:200]}...")
            except Exception as e:
                print(f"   ❌ Direct vision test failed: {e}")
        else:
            print("   ❌ No vision analyzer available")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_vision_command())

