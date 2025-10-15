#!/usr/bin/env python3
"""
Test JARVIS Document Writer with Real Claude API
"""

import asyncio
import json
import os
from pathlib import Path
import sys

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_document_writer():
    """Test the complete document writer flow"""
    from api.unified_command_processor import UnifiedCommandProcessor

    # Create processor
    processor = UnifiedCommandProcessor()

    # Test command
    command = "Write me a short essay about renewable energy"

    print("=" * 60)
    print("Testing JARVIS Document Writer")
    print("=" * 60)
    print(f"\nCommand: {command}")
    print("\nProcessing...")

    # Process command (without websocket)
    result = await processor.process_command(command, websocket=None)

    print("\nResult:")
    print(json.dumps(result, indent=2))

    # Check if it was successful
    if result.get('success'):
        print("\n✅ Document creation initiated successfully!")

        # Check if we're using real API or mock
        if result.get('document_id'):
            print(f"Document ID: {result['document_id']}")
        if result.get('document_url'):
            print(f"Document URL: {result['document_url']}")
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")

    return result


async def test_claude_api_status():
    """Check Claude API status"""
    print("\n" + "=" * 60)
    print("Checking Claude API Status")
    print("=" * 60)

    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        print(f"✅ API Key found: {api_key[:20]}...")

        # Test the key
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=20,
                messages=[{"role": "user", "content": "Say 'API working'"}]
            )
            print(f"✅ API Test: {response.content[0].text}")
            return True
        except Exception as e:
            print(f"❌ API Test Failed: {e}")
            return False
    else:
        print("❌ No API Key found in environment")
        return False


async def main():
    """Run tests"""
    # Check API first
    api_working = await test_claude_api_status()

    if api_working:
        print("\n🚀 Claude API is configured correctly!")
        print("The document writer will use REAL AI content generation.")
    else:
        print("\n⚠️ Claude API not configured")
        print("The document writer will use DEMO content.")

    # Test document writer
    print("\n" + "=" * 60)
    input("Press Enter to test document writer...")

    result = await test_document_writer()

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())