#!/usr/bin/env python3
"""
Test Document Writer Integration
=================================

Simple test to verify the document writer components work together.
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))


async def test_intent_analysis():
    """Test intent analyzer recognizes document creation commands"""
    print("\n=== Testing Intent Analysis ===")

    # Import directly to avoid __init__.py issues
    sys.path.insert(0, str(Path(__file__).parent / "context_intelligence" / "analyzers"))
    from intent_analyzer import IntentAnalyzer

    analyzer = IntentAnalyzer()

    test_commands = [
        "write me an essay about climate change",
        "create a 500 word report on AI",
        "draft a blog post about Python programming",
        "help me write a paper on quantum computing"
    ]

    for command in test_commands:
        intent = await analyzer.analyze(command)
        print(f"\nCommand: {command}")
        print(f"  Intent Type: {intent.type.value}")
        print(f"  Confidence: {intent.confidence}")
        print(f"  Requires Screen: {intent.requires_screen}")
        print(f"  Entities: {intent.entities}")

        # Verify it's recognized as document creation
        assert intent.type.value == "document_creation", f"Failed to recognize: {command}"

    print("\n✅ Intent analysis test passed!")


async def test_document_request_parsing():
    """Test parsing of document requests"""
    print("\n=== Testing Document Request Parsing ===")

    sys.path.insert(0, str(Path(__file__).parent / "context_intelligence" / "executors"))
    from document_writer import parse_document_request

    test_cases = [
        ("write me a 500 word essay about climate change", {
            "expected_type": "essay",
            "expected_word_count": 500,
            "expected_topic_contains": "climate"
        }),
        ("create a report on artificial intelligence", {
            "expected_type": "report",
            "expected_word_count": None,
            "expected_topic_contains": "artificial"
        }),
        ("draft a 3 page paper on quantum computing", {
            "expected_type": "paper",
            "expected_page_count": 3,
            "expected_topic_contains": "quantum"
        }),
    ]

    for command, expectations in test_cases:
        request = parse_document_request(command, {})

        print(f"\nCommand: {command}")
        print(f"  Document Type: {request.document_type.value}")
        print(f"  Topic: {request.topic}")
        print(f"  Word Count: {request.word_count}")
        print(f"  Page Count: {request.page_count}")
        print(f"  Platform: {request.platform.value}")

        # Verify expectations
        assert request.document_type.value == expectations["expected_type"]
        assert request.word_count == expectations.get("expected_word_count")
        assert request.page_count == expectations.get("expected_page_count")

        if "expected_topic_contains" in expectations:
            assert expectations["expected_topic_contains"].lower() in request.topic.lower()

    print("\n✅ Document request parsing test passed!")


async def test_claude_streamer():
    """Test Claude content streamer (mock mode)"""
    print("\n=== Testing Claude Streamer ===")

    sys.path.insert(0, str(Path(__file__).parent / "context_intelligence" / "automation"))
    from claude_streamer import ClaudeContentStreamer

    streamer = ClaudeContentStreamer()

    # Test outline generation
    print("\nGenerating outline...")
    outline = await streamer.generate_outline(
        "Create an outline for an essay about climate change"
    )

    print(f"  Title: {outline['title']}")
    print(f"  Sections: {len(outline['sections'])}")
    for section in outline['sections']:
        print(f"    - {section['name']}")

    # Test content streaming
    print("\nTesting content streaming...")
    word_count = 0
    chunk_count = 0

    async for chunk in streamer.stream_content(
        "Write a short paragraph about Python programming",
        max_tokens=100
    ):
        word_count += len(chunk.split())
        chunk_count += 1

    print(f"  Chunks received: {chunk_count}")
    print(f"  Total words: {word_count}")

    assert chunk_count > 0, "No chunks received from streamer"
    assert word_count > 10, "Not enough content generated"

    print("\n✅ Claude streamer test passed!")


async def test_browser_controller():
    """Test browser controller (without actually opening browser)"""
    print("\n=== Testing Browser Controller ===")

    sys.path.insert(0, str(Path(__file__).parent / "context_intelligence" / "automation"))
    from browser_controller import BrowserController

    controller = BrowserController()

    print(f"  Browser: {controller.browser}")
    print(f"  Controller initialized: ✓")

    # Test script building (without executing)
    script = controller._build_navigation_script("https://docs.google.com/document/create")
    assert "Safari" in script or "Chrome" in script
    assert "docs.google.com" in script

    print("  Navigation script generation: ✓")

    # Test JS escaping
    escaped = controller._escape_js('var x = "test";')
    assert '"' not in escaped or '\\"' in escaped

    print("  JavaScript escaping: ✓")

    print("\n✅ Browser controller test passed!")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("Document Writer Integration Tests")
    print("=" * 60)

    try:
        await test_intent_analysis()
        await test_document_request_parsing()
        await test_claude_streamer()
        await test_browser_controller()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)

        print("\n📝 Document Writer is ready for integration.")
        print("\nTo use it, say something like:")
        print('  - "Write me an essay about climate change"')
        print('  - "Create a 500 word report on AI"')
        print('  - "Draft a blog post about Python"')

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
