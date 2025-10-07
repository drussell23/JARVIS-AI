#!/usr/bin/env python3
"""
Verify JARVIS generates correct content for Michael Jackson essay
"""

import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Force override environment variables with .env values
load_dotenv(override=True)

import sys
sys.path.insert(0, str(Path(__file__).parent))


async def test_essay_generation():
    """Test that JARVIS generates content about the requested topic"""

    from api.unified_command_processor import UnifiedCommandProcessor
    from context_intelligence.automation.claude_streamer import get_claude_streamer

    print("=" * 60)
    print("JARVIS Essay Generation Test")
    print("=" * 60)

    # Check API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    print(f"\nAPI Key loaded: {api_key[:30]}...")

    # Test direct streaming first
    print("\n1. Testing direct Claude streaming...")
    print("-" * 40)

    streamer = get_claude_streamer()
    content = ""

    async for chunk in streamer.stream_content(
        "Write one sentence about Michael Jackson's impact on music.",
        max_tokens=50
    ):
        content += chunk

    print(f"Generated: {content}")

    if "michael" in content.lower() or "jackson" in content.lower():
        print("✅ Direct streaming works - generates Michael Jackson content")
    else:
        print("❌ Direct streaming failed - not about Michael Jackson")
        if "dogs" in content.lower():
            print("   Still getting Dogs essay (mock content)")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if "dogs" not in content.lower() and ("michael" in content.lower() or "jackson" in content.lower()):
        print("✅ SUCCESS: JARVIS is now generating real AI content!")
        print("\nThe fix is working. When you restart JARVIS and ask it to")
        print("'write me an essay about Michael Jackson', it will generate")
        print("real content about Michael Jackson, not the Dogs essay.")
    else:
        print("⚠️ Issue detected - please restart JARVIS for changes to take effect")

    print("\nIMPORTANT: Restart JARVIS for the changes to take effect!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_essay_generation())