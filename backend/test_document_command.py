#!/usr/bin/env python3
"""
Test document command classification and routing
"""

import asyncio
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_document_command():
    """Test if 'write me an essay on dogs' gets classified correctly"""

    # Import the unified command processor
    from api.unified_command_processor import UnifiedCommandProcessor

    # Create processor
    processor = UnifiedCommandProcessor()

    # Test commands
    test_commands = [
        "write me an essay on dogs",
        "write an essay about dogs",
        "create an essay on dogs",
        "write me a report on climate change",
        "draft a paper about artificial intelligence"
    ]

    print("\n" + "="*80)
    print("TESTING DOCUMENT COMMAND CLASSIFICATION")
    print("="*80 + "\n")

    for command in test_commands:
        print(f"\n📝 Testing: '{command}'")
        print("-" * 80)

        # Classify the command
        command_type, confidence = await processor._classify_command(command)

        print(f"   Result: {command_type.value}")
        print(f"   Confidence: {confidence}")

        if command_type.value == 'document':
            print(f"   ✅ CORRECT - Would route to document writer")

            # Test parsing
            from context_intelligence.executors import parse_document_request
            doc_request = parse_document_request(command, {})

            print(f"\n   Parsed Request:")
            print(f"      Topic: {doc_request.topic}")
            print(f"      Type: {doc_request.document_type.value}")
            print(f"      Format: {doc_request.formatting.value}")
            print(f"      Title: {doc_request.title}")
            print(f"      Length: {doc_request.get_length_spec()}")
        else:
            print(f"   ❌ WRONG - Got {command_type.value} instead of document")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    asyncio.run(test_document_command())
