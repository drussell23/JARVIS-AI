#!/usr/bin/env python3
"""
Test conversational responses for browser commands
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.unified_command_processor import get_unified_processor

async def test_responses():
    processor = get_unified_processor()
    
    test_commands = [
        "search for wells fargo",
        "search for python tutorials",
        "open safari",
        "open chrome and search for weather",
        "google machine learning",
        "go to github",
    ]
    
    print("Testing conversational responses:")
    print("=" * 60)
    
    for command in test_commands:
        print(f"\nYou: {command}")
        result = await processor.process_command(command)
        print(f"JARVIS: {result.get('response')}")

if __name__ == "__main__":
    asyncio.run(test_responses())