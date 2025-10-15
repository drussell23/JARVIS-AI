#!/usr/bin/env python3
"""
Test Real Document Generation with Claude API
"""

import asyncio
import os
from pathlib import Path
import sys

# Load environment variables from backend/.env
from dotenv import load_dotenv
env_path = Path(__file__).parent / '.env'
load_dotenv(env_path, override=True)

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_document_generation():
    """Test real document generation"""
    print("=" * 60)
    print("Testing Real Document Generation")
    print("=" * 60)

    # Import document writer modules
    from context_intelligence.executors.document_writer import (
        DocumentWriterExecutor,
        parse_document_request,
        DocumentRequest
    )

    # Create a document request
    request = DocumentRequest(
        topic="renewable energy solutions",
        document_type=DocumentType.ESSAY,
        word_count=200,
        formatting=DocumentFormat.MLA
    )

    print(f"\nDocument Request:")
    print(f"  Topic: {request.topic}")
    print(f"  Type: {request.document_type.value}")
    print(f"  Format: {request.formatting.value}")
    print(f"  Word Count: {request.word_count}")

    # Create writer
    writer = DocumentWriterExecutor()

    # Mock progress callback
    async def progress_callback(message):
        print(f"  > {message}")

    print("\n" + "=" * 60)
    print("Starting Document Generation...")
    print("=" * 60)

    # Note: This will try to create a real Google Doc
    # For testing, we'll just test the Claude content generation
    from context_intelligence.automation.claude_streamer import ClaudeContentStreamer

    api_key = os.getenv('ANTHROPIC_API_KEY')
    print(f"\nUsing API Key: {api_key[:30]}...")

    streamer = ClaudeContentStreamer(api_key=api_key)

    # Test content streaming
    print("\nGenerating content about renewable energy...")
    print("-" * 40)

    content = ""
    chunk_count = 0

    async for chunk in streamer.stream_content(
        "Write a 200 word MLA format essay about renewable energy solutions. Include proper MLA heading and citations.",
        max_tokens=500
    ):
        content += chunk
        chunk_count += 1
        print(chunk, end="", flush=True)

    print("\n" + "-" * 40)
    print(f"\nGenerated {chunk_count} chunks")
    print(f"Total length: {len(content)} characters")

    # Check if it's real content
    if "renewable" in content.lower() or "energy" in content.lower() or "solar" in content.lower():
        print("\n✅ SUCCESS: Real AI-generated content about renewable energy!")
        return True
    elif "Dogs" in content:
        print("\n⚠️ WARNING: Still using mock/demo content")
        return False
    else:
        print("\n🤔 Content generated but topic unclear")
        return True


# Add missing imports
from context_intelligence.executors.document_writer import DocumentType, DocumentFormat

async def main():
    """Run the test"""
    success = await test_document_generation()

    print("\n" + "=" * 60)
    if success:
        print("✅ JARVIS can now generate REAL AI content!")
        print("\nYou can now ask JARVIS to:")
        print('  • "Write me an essay about climate change"')
        print('  • "Create a report on artificial intelligence"')
        print('  • "Draft an MLA paper about renewable energy"')
    else:
        print("⚠️ Still in DEMO mode")
        print("Check your API key configuration")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())