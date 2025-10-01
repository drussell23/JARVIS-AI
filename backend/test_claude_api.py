#!/usr/bin/env python3
"""
Test Claude API Connection
===========================

Test script to verify Claude API is working correctly.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_claude_api_direct():
    """Test Claude API directly"""
    print("\n=== Testing Direct Claude API ===")

    try:
        import anthropic

        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ No ANTHROPIC_API_KEY found in environment")
            return False

        print(f"✓ API Key found: {api_key[:8]}...")

        client = anthropic.Anthropic(api_key=api_key)
        print("✓ Anthropic client created")

        # Test a simple completion
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": "Say 'Hello from Claude API' and nothing else."
            }]
        )

        print(f"✓ Response: {response.content[0].text}")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_claude_streamer():
    """Test Claude streamer with API key"""
    print("\n=== Testing Claude Streamer ===")

    sys.path.insert(0, str(Path(__file__).parent / "context_intelligence" / "automation"))
    from claude_streamer import ClaudeContentStreamer

    api_key = os.getenv('ANTHROPIC_API_KEY')
    print(f"API Key from env: {api_key[:8] if api_key else 'None'}...")

    streamer = ClaudeContentStreamer(api_key=api_key)

    # Check if client is initialized
    print(f"Streamer client: {streamer._client}")
    print(f"Streamer API key: {streamer.api_key[:8] if streamer.api_key else 'None'}...")

    # Test streaming
    print("\nTesting real streaming...")
    content = ""
    chunk_count = 0

    try:
        async for chunk in streamer.stream_content(
            "Write a single sentence about dogs.",
            max_tokens=50,
            model="claude-3-5-sonnet-20241022"
        ):
            content += chunk
            chunk_count += 1
            print(f"Chunk {chunk_count}: {chunk}")
    except Exception as e:
        print(f"Error during streaming: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nTotal chunks: {chunk_count}")
    print(f"Total content: {content}")

    # Check if it's mock content
    if "Dogs have been humanity's faithful companions" in content:
        print("⚠️  This appears to be MOCK content, not real Claude API")
        return False
    else:
        print("✓ This appears to be REAL Claude API content")
        return True


async def main():
    """Run tests"""
    print("=" * 60)
    print("Claude API Connection Test")
    print("=" * 60)

    api_direct_works = await test_claude_api_direct()
    streamer_works = await test_claude_streamer()

    print("\n" + "=" * 60)
    print("Results:")
    print(f"  Direct API: {'✓ Working' if api_direct_works else '❌ Failed'}")
    print(f"  Streamer: {'✓ Working' if streamer_works else '❌ Failed'}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())