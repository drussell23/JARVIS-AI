#!/usr/bin/env python3
"""
Test various browser automation commands
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.unified_command_processor import get_unified_processor

async def test_variations():
    processor = get_unified_processor()
    
    test_commands = [
        "open safari and search for dogs",
        "open chrome and search for python tutorials",
        "search for cats",  # Should use default or context browser
        "open a new tab and search for weather",
        "google artificial intelligence",
        "open firefox and go to github",
    ]
    
    print("Testing browser command variations:")
    print("=" * 60)
    
    for command in test_commands:
        print(f"\nCommand: '{command}'")
        result = await processor.process_command(command)
        print(f"Response: {result.get('response')}")
        if not result.get('success'):
            print(f"Error: {result.get('error', 'Command failed')}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")

if __name__ == "__main__":
    asyncio.run(test_variations())