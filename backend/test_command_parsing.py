#!/usr/bin/env python3
"""
Test command parsing for browser automation
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.unified_command_processor import get_unified_processor

async def test_parsing():
    processor = get_unified_processor()
    
    test_commands = [
        "open safari and search for dogs",
        "open chrome and go to google",
        "open safari and go to github and search for python",
        "open a new tab and type hello world",
        "go to google.com and search for cats"
    ]
    
    print("Testing command parsing:\n")
    
    for command in test_commands:
        print(f"Command: '{command}'")
        
        # Test classification
        cmd_type, confidence = await processor._classify_command(command)
        print(f"  Type: {cmd_type.value} (confidence: {confidence})")
        
        # Test parsing if compound
        if cmd_type.value == "compound":
            parts = processor._parse_compound_parts(command)
            print(f"  Parts: {parts}")
        
        print()

if __name__ == "__main__":
    asyncio.run(test_parsing())