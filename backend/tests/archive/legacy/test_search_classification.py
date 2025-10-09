#!/usr/bin/env python3
"""
Test why "search for dogs" isn't being classified as system
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.unified_command_processor import get_unified_processor

async def test_classification():
    processor = get_unified_processor()
    
    test_commands = [
        "search for dogs",
        "google dogs", 
        "search dogs",
        "look up dogs",
        "find dogs online"
    ]
    
    print("Testing search command classification:\n")
    
    for command in test_commands:
        cmd_type, confidence = await processor._classify_command(command)
        print(f"'{command}' -> {cmd_type.value} (confidence: {confidence})")
        
        # Check system patterns
        command_lower = command.lower()
        system_patterns = [
            'open', 'close', 'launch', 'quit', 'start', 'restart', 'shutdown',
            'volume', 'brightness', 'settings', 'wifi', 'wi-fi', 'screenshot',
            'mute', 'unmute', 'sleep display', 'go to', 'navigate to', 'visit',
            'browse to', 'search for', 'google', 'new tab', 'open tab', 'type',
            'enter', 'search bar', 'click', 'another tab', 'open another'
        ]
        
        matches = [p for p in system_patterns if p in command_lower]
        if matches:
            print(f"  -> Matched patterns: {matches}")
        print()

if __name__ == "__main__":
    asyncio.run(test_classification())